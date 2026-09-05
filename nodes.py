"""
Video Inpaint Crop / Stitch  —  written for Romano, 2026-09-01.

Why this exists
---------------
Neither existing node works for a moving subject in video:

  comfyui-inpaint-cropandstitch : recomputes the crop box PER FRAME, so the
      subject is re-normalised every frame.  Head scale drifts, and its
      `context_from_mask_extend_factor` is a MULTIPLIER, so asking for margin
      also enlarges the head.  Margin and size are the same dial.

  ComfyUI-Pixaroma              : takes frame 0's mask only and uses ONE fixed
      rect for the whole batch (`m = m[0]`, "all batch frames, same rect").
      Additive margin, which is right, but the box never follows the subject.

This pair does both correct things at once:

  * CONSTANT box SIZE across the batch  -> the head renders at the same scale
    in every frame.  No renormalising, no drift.
  * MOVING box CENTRE, per frame        -> the box follows the subject through
    a turn or a walk.
  * ADDITIVE margin in pixels           -> margin no longer costs head size.
"""

import torch
import torch.nn.functional as F
import comfy.utils

CROP_INFO = "VIDEO_CROP_INFO"


def _bboxes(mask, thresh):
    """Per-frame (y0, y1, x0, x1) of the mask, or None where the frame is empty."""
    m = mask > thresh
    any_y = m.any(dim=2)          # [B, H]
    any_x = m.any(dim=1)          # [B, W]
    out = []
    for b in range(m.shape[0]):
        ys = torch.nonzero(any_y[b], as_tuple=True)[0]
        xs = torch.nonzero(any_x[b], as_tuple=True)[0]
        if len(ys) == 0 or len(xs) == 0:
            out.append(None)
        else:
            out.append((int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])))
    return out


def _round_up(v, mult):
    return int(((int(v) + mult - 1) // mult) * mult)


class VideoInpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_px": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Margin around the mask, in ORIGINAL-frame pixels. Additive: "
                               "it does NOT change how big the subject renders."}),
                "size_mode": (["largest frame", "median frame", "percentile"], {"default": "largest frame",
                    "tooltip": "How the one constant box size is chosen across the batch."}),
                "percentile": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only when size_mode is 'percentile'."}),
                "size_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01,
                    "tooltip": "Scales the box. BELOW 1.0 makes the subject render SMALLER. "
                               "This is the head-size dial, and it is independent of context_px."}),
                "square": ("BOOLEAN", {"default": True,
                    "tooltip": "Force a square box (what LTX crops usually want)."}),
                "multiple": ([8, 16, 32, 64], {"default": 32,
                    "tooltip": "Round the box up to a multiple of this. LTX's VAE needs 32."}),
                "target": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Resolution the crop is resized to for sampling."}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smooth_centre": ("INT", {"default": 5, "min": 1, "max": 61, "step": 2,
                    "tooltip": "Moving-average window over the box centre, in frames. Odd number. "
                               "Higher = steadier box, less jitter. 1 disables smoothing."}),
                "upscale_algorithm": (["lanczos", "bicubic", "bilinear", "nearest-exact", "area"],
                                      {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", CROP_INFO, "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_info", "width", "height")
    FUNCTION = "run"
    CATEGORY = "video/inpaint"
    DESCRIPTION = ("Crop a moving subject out of a video for inpainting. Uses ONE box size for the "
                   "whole clip (so scale never drifts) but moves the box per frame (so it follows "
                   "the subject). Margin is additive in pixels, so it never changes subject size.")

    def run(self, image, mask, context_px, size_mode, percentile, size_scale, square,
            multiple, target, mask_threshold, smooth_centre, upscale_algorithm):
        B, H, W, C = image.shape
        if mask.shape[0] != B:
            mask = comfy.utils.repeat_to_batch_size(mask, B)
        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="bilinear",
                             align_corners=False).squeeze(1) if mask.shape[1:] != (H, W) else mask

        boxes = _bboxes(mask, mask_threshold)
        valid = [b for b in boxes if b is not None]
        if not valid:
            raise ValueError("VideoInpaintCrop: the mask is empty on every frame.")

        # ---- one constant box SIZE for the whole clip -------------------------
        hs = torch.tensor([b[1] - b[0] + 1 for b in valid], dtype=torch.float32)
        ws = torch.tensor([b[3] - b[2] + 1 for b in valid], dtype=torch.float32)
        if size_mode == "largest frame":
            bh, bw = hs.max().item(), ws.max().item()
        elif size_mode == "median frame":
            bh, bw = hs.median().item(), ws.median().item()
        else:
            q = max(0.0, min(1.0, percentile))
            bh, bw = torch.quantile(hs, q).item(), torch.quantile(ws, q).item()

        bh = bh * size_scale + 2 * context_px
        bw = bw * size_scale + 2 * context_px
        if square:
            bh = bw = max(bh, bw)
        bh = _round_up(bh, multiple)
        bw = _round_up(bw, multiple)
        if bh > H:
            bh = max(multiple, (H // multiple) * multiple)   # round DOWN to fit
        if bw > W:
            bw = max(multiple, (W // multiple) * multiple)

        # ---- per-frame CENTRES, carried through empty frames, then smoothed ---
        cy, cx, last = [], [], None
        for b in boxes:
            if b is None:
                b = last if last is not None else valid[0]
            last = b
            cy.append((b[0] + b[1]) / 2.0)
            cx.append((b[2] + b[3]) / 2.0)
        cy = torch.tensor(cy, dtype=torch.float32)
        cx = torch.tensor(cx, dtype=torch.float32)

        k = int(smooth_centre) | 1
        if k > 1 and B > 1:
            pad = k // 2
            ker = torch.ones(1, 1, k) / k
            cy = F.conv1d(F.pad(cy.view(1, 1, -1), (pad, pad), mode="replicate"), ker).view(-1)
            cx = F.conv1d(F.pad(cx.view(1, 1, -1), (pad, pad), mode="replicate"), ker).view(-1)

        # ---- cut every frame with the same-sized box, clamped to the image ----
        imgs, masks, rects = [], [], []
        for b in range(B):
            y0 = int(round(cy[b].item() - bh / 2.0))
            x0 = int(round(cx[b].item() - bw / 2.0))
            y0 = max(0, min(y0, H - bh))
            x0 = max(0, min(x0, W - bw))
            rects.append((y0, x0, bh, bw))
            imgs.append(image[b:b + 1, y0:y0 + bh, x0:x0 + bw, :])
            masks.append(mask[b:b + 1, y0:y0 + bh, x0:x0 + bw])

        # Resize for sampling. A square box maps to target x target. A non-square
        # box maps target to its LONG side and scales the short side to preserve
        # aspect (no distortion), rounded to /8 so latents stay valid.
        if square or bh == bw:
            tw = th = target
        elif bw >= bh:
            tw = target
            th = max(8, int(round(target * bh / bw / 8.0)) * 8)
        else:
            th = target
            tw = max(8, int(round(target * bw / bh / 8.0)) * 8)

        ci = torch.cat(imgs, 0).movedim(-1, 1)
        ci = comfy.utils.common_upscale(ci, tw, th, upscale_algorithm, "disabled").movedim(1, -1)
        cm = F.interpolate(torch.cat(masks, 0).unsqueeze(1), size=(th, tw),
                           mode="bilinear", align_corners=False).squeeze(1).clamp(0, 1)

        info = {"rects": rects, "orig_h": H, "orig_w": W, "box_h": bh, "box_w": bw,
                "target_w": tw, "target_h": th, "algorithm": upscale_algorithm}
        return (ci, cm, info, tw, th)


class VideoInpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "crop_info": (CROP_INFO,),
                "blend_px": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Feather width at the paste-back seam, in cropped pixels."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Cropped mask. If wired, only the masked area is "
                                             "pasted back; otherwise the whole box is."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "video/inpaint"
    DESCRIPTION = "Paste the inpainted crops back at each frame's own rect, feathered at the seam."

    def run(self, original_image, inpainted_image, crop_info, blend_px, mask=None):
        rects = crop_info["rects"]
        out = original_image.clone()
        B = out.shape[0]
        n = min(B, inpainted_image.shape[0], len(rects))

        for b in range(n):
            y0, x0, bh, bw = rects[b]
            patch = inpainted_image[b:b + 1].movedim(-1, 1)
            patch = comfy.utils.common_upscale(patch, bw, bh, crop_info.get("algorithm", "lanczos"),
                                               "disabled").movedim(1, -1)
            if mask is not None:
                m = mask[min(b, mask.shape[0] - 1)].unsqueeze(0).unsqueeze(0)
                m = F.interpolate(m, size=(bh, bw), mode="bilinear", align_corners=False)
                if blend_px > 0:
                    k = int(blend_px) | 1
                    ker = torch.ones(1, 1, k, k, device=m.device) / (k * k)
                    m = F.conv2d(F.pad(m, (k // 2,) * 4, mode="replicate"), ker)
                m = m.squeeze(1).squeeze(0).unsqueeze(-1).clamp(0, 1)
            else:
                m = torch.ones(bh, bw, 1)
                if blend_px > 0:
                    r = torch.linspace(0, 1, min(blend_px, bh // 2, bw // 2) or 1)
                    e = len(r)
                    m[:e, :, 0] *= r.view(-1, 1); m[-e:, :, 0] *= r.flip(0).view(-1, 1)
                    m[:, :e, 0] *= r.view(1, -1); m[:, -e:, 0] *= r.flip(0).view(1, -1)
            m = m.to(out.device, out.dtype)
            base = out[b, y0:y0 + bh, x0:x0 + bw, :]
            out[b, y0:y0 + bh, x0:x0 + bw, :] = m * patch[0].to(out.device, out.dtype) + (1 - m) * base

        return (out,)


class MaskHoldLast:
    """When a tracker drops frames (fast rotation, occlusion), empty masks punch
    holes in the fill. This node carries the last non-empty mask forward through
    the gap - and back-fills leading empties from the first good one - so the
    marker never blinks out mid-clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_coverage": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "A frame counts as EMPTY if its mask covers less than this fraction of pixels."}),
            },
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("mask", "frames_held")
    FUNCTION = "run"
    CATEGORY = "video/inpaint"
    DESCRIPTION = "Replace empty tracker frames with the last valid mask (back-fills leading gaps)."

    def run(self, mask, min_coverage):
        out = mask.clone()
        B = out.shape[0]
        area = out.shape[1] * out.shape[2]
        valid = [(out[b].sum().item() / area) >= min_coverage for b in range(B)]
        held = 0
        last = None
        for b in range(B):
            if valid[b]:
                last = b
            elif last is not None:
                out[b] = out[last]; held += 1
        nxt = None
        for b in range(B - 1, -1, -1):
            if valid[b]:
                nxt = b
            elif last is None or b < (valid.index(True) if any(valid) else 0):
                if nxt is not None:
                    out[b] = out[nxt]; held += 1
        return (out, held)
