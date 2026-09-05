# ComfyUI RomanoNodes

ComfyUI nodes for video inpainting on a moving subject: Video Inpaint Crop,
Video Inpaint Stitch, and Mask Hold Last.

Crop a **moving** subject out of a video for inpainting, render it at full
resolution, and paste it back.

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
