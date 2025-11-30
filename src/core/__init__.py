"""Core modules for Sora watermark remover."""

from .audio_preservation import AudioPreservationPipeline
from .yolo_detector import YOLOWorldDetector
from .sam3_segmenter import SAM3Segmenter, SAM3SegmenterLocal
from .propainter import ProPainterInpainter, ProPainterInpainterLite
from .mask_utils import (
    expand_bbox,
    dilate_mask,
    erode_mask,
    merge_masks,
    create_empty_mask,
    bbox_to_mask,
    save_mask,
    load_mask,
    get_mask_coverage,
    smooth_mask_edges,
    validate_mask
)

__all__ = [
    # Audio
    "AudioPreservationPipeline",
    # Detection
    "YOLOWorldDetector",
    # Segmentation
    "SAM3Segmenter",
    "SAM3SegmenterLocal",
    # Inpainting
    "ProPainterInpainter",
    "ProPainterInpainterLite",
    # Mask utilities
    "expand_bbox",
    "dilate_mask",
    "erode_mask",
    "merge_masks",
    "create_empty_mask",
    "bbox_to_mask",
    "save_mask",
    "load_mask",
    "get_mask_coverage",
    "smooth_mask_edges",
    "validate_mask",
]
