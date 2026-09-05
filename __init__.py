from .nodes import VideoInpaintCrop, VideoInpaintStitch, MaskHoldLast

NODE_CLASS_MAPPINGS = {
    "VideoInpaintCrop": VideoInpaintCrop,
    "VideoInpaintStitch": VideoInpaintStitch,
    "MaskHoldLast": MaskHoldLast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoInpaintCrop": "Video Inpaint Crop (stable size, moving box)",
    "VideoInpaintStitch": "Video Inpaint Stitch",
    "MaskHoldLast": "Mask Hold Last (tracker dropout fix)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
