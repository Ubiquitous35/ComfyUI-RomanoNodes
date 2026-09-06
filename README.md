# ComfyUI RomanoNodes

ComfyUI nodes for inpaint crop-and-stitch on images **and** video — including
the hard case no other pack handles: a moving subject. Video Inpaint Crop,
Video Inpaint Stitch, and Mask Hold Last.

Crop a masked subject out of a still image or video for inpainting, render it
at full resolution, and paste it back. A single image is just a one-frame
clip — the same two nodes do both.

## Why

Neither existing node works for a moving subject:

| node | box size | box position | margin |
|---|---|---|---|
| comfyui-inpaint-cropandstitch | recomputed **per frame** | follows subject | **multiplier** — margin also enlarges the subject |
| ComfyUI-Pixaroma | one **fixed** box | frame 0 only, never moves | additive (correct) |

Per-frame sizing renormalises the subject on every frame, so its scale drifts.
A fixed box does not follow the subject, so it lands in the wrong place. And a
multiplicative margin means you cannot ask for edge clearance without also
making the subject bigger.

## What this does

- **One constant box size** for the whole clip → the subject renders at the same
  scale in every frame.
- **Box centre moves per frame** → it follows a turn or a walk.
- **`context_px` is additive** → margin and subject size are independent dials.
- **`size_scale`** is the subject-size dial. Below 1.0 renders it smaller.
- **`smooth_centre`** applies a moving average over the box centre so a jittery
  mask does not shake the crop.

## Mask Hold Last (tracker dropout fix)

When a tracker drops frames on a fast turn or occlusion, empty masks punch holes
in the fill. This node carries the last valid mask forward through the gap (and
back-fills leading empties), so the mask never blinks out mid-clip.

## Wiring

```
image ─┐
mask  ─┴→ Video Inpaint Crop ─→ cropped_image → (your sampler)
                              ─→ cropped_mask  → (mask processing)
                              ─→ crop_info ────┐
original image ──────────────────────────────┐ │
sampler output ──────────────────────────────┴─┴→ Video Inpaint Stitch → image
```

`multiple` defaults to 32 because LTX's VAE requires dimensions divisible by 32.

## Verified capability

Verified on both single still images and moving video. The same Crop/Stitch pair
handles a single image as a one-frame batch, and maintains stable crop geometry
through moving and rotating video subjects — including a full front-to-rear
subject rotation and a second-pass refine where the stitch rescaled a 1.5×
patch back into the original frames.

## Tested performance

Tested on: NVIDIA RTX PRO 6000 Blackwell 96 GB, AMD Ryzen 9 9950X3D, 192 GB RAM,
Ubuntu / ComfyUI.

Example full markerless 2-pass video inpaint run built on these nodes:

- 193 input frames (~7.5 s), 576×1024 working resolution
- Pass 1: 8 LTX steps, ~7–8 min
- Pass 2: 3 high-resolution refine steps, ~4–5 min
- Full fresh run: ~16–16.5 min
- SAM3 tracking: ~10 s for 193 frames
- Crop and Stitch themselves are seconds of that total — the cost is diffusion

Results will vary by model, resolution, frame count, sampler, LoRAs, and hardware.
