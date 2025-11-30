#!/usr/bin/env python3
"""
Mask Utilities for Watermark Detection Pipeline

Handles:
- Bounding box expansion (to capture glow/animation effects)
- Mask dilation (ensure complete coverage)
- Multi-mask merging (SORA + @username → single mask)
- Mask I/O for ProPainter compatibility
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union


def expand_bbox(
    bbox: List[float],
    expand_px: int,
    frame_shape: Tuple[int, int, int]
) -> List[int]:
    """
    Expand bounding box to capture glow/animation effects around watermark.

    Args:
        bbox: [x1, y1, x2, y2] coordinates
        expand_px: Pixels to expand in each direction
        frame_shape: (height, width, channels) of the frame

    Returns:
        Expanded [x1, y1, x2, y2] clipped to frame bounds
    """
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]

    # Expand and clip to frame bounds
    x1 = max(0, int(x1) - expand_px)
    y1 = max(0, int(y1) - expand_px)
    x2 = min(width, int(x2) + expand_px)
    y2 = min(height, int(y2) + expand_px)

    return [x1, y1, x2, y2]


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    """
    Dilate mask to ensure complete watermark coverage.

    Prevents edge artifacts by expanding mask boundaries.

    Args:
        mask: Binary mask (0/255 or 0/1)
        kernel_size: Size of dilation kernel (must be odd)
        iterations: Number of dilation passes

    Returns:
        Dilated binary mask (same dtype as input)
    """
    if mask is None or mask.size == 0:
        return mask

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # Normalize to 0-255 for cv2 operations
    was_normalized = mask.max() <= 1
    if was_normalized:
        mask = (mask * 255).astype(np.uint8)

    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    # Restore original scale
    if was_normalized:
        dilated = (dilated / 255).astype(np.float32)

    return dilated


def erode_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Erode mask to clean up noise.

    Args:
        mask: Binary mask (0/255 or 0/1)
        kernel_size: Size of erosion kernel (must be odd)
        iterations: Number of erosion passes

    Returns:
        Eroded binary mask
    """
    if mask is None or mask.size == 0:
        return mask

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    was_normalized = mask.max() <= 1
    if was_normalized:
        mask = (mask * 255).astype(np.uint8)

    eroded = cv2.erode(mask, kernel, iterations=iterations)

    if was_normalized:
        eroded = (eroded / 255).astype(np.float32)

    return eroded


def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Merge multiple masks (e.g., SORA + @username) into single mask.

    Uses logical OR - any pixel covered by any mask is included.

    Args:
        masks: List of binary masks (same shape)

    Returns:
        Merged binary mask (uint8, 0-255)
    """
    if not masks:
        raise ValueError("No masks provided to merge")

    # Start with first mask
    merged = masks[0].copy()

    # Normalize to 0-255
    if merged.max() <= 1:
        merged = (merged * 255).astype(np.uint8)
    else:
        merged = merged.astype(np.uint8)

    # OR each subsequent mask
    for mask in masks[1:]:
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        merged = cv2.bitwise_or(merged, mask)

    return merged


def create_empty_mask(frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create an empty (all zeros) mask matching frame dimensions.

    Args:
        frame_shape: (height, width, channels) of the frame

    Returns:
        Empty uint8 mask (height, width)
    """
    height, width = frame_shape[:2]
    return np.zeros((height, width), dtype=np.uint8)


def bbox_to_mask(
    bbox: List[int],
    frame_shape: Tuple[int, int, int],
    fill_value: int = 255
) -> np.ndarray:
    """
    Create a rectangular mask from a bounding box.

    Useful as fallback when SAM segmentation fails.

    Args:
        bbox: [x1, y1, x2, y2] coordinates
        frame_shape: (height, width, channels) of the frame
        fill_value: Value to fill the bbox region

    Returns:
        Binary mask with bbox region filled
    """
    mask = create_empty_mask(frame_shape)
    x1, y1, x2, y2 = [int(c) for c in bbox]
    mask[y1:y2, x1:x2] = fill_value
    return mask


def save_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    as_binary: bool = True
) -> Path:
    """
    Save mask to file (PNG format for lossless).

    Args:
        mask: Binary mask to save
        output_path: Output file path
        as_binary: If True, ensure 0/255 values only

    Returns:
        Path to saved mask file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if as_binary:
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    cv2.imwrite(str(output_path), mask)
    return output_path


def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load mask from file.

    Args:
        mask_path: Path to mask file

    Returns:
        Loaded mask as uint8 array
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    return mask


def get_mask_coverage(mask: np.ndarray) -> float:
    """
    Calculate percentage of frame covered by mask.

    Useful for validation (watermarks should be <10% typically).

    Args:
        mask: Binary mask

    Returns:
        Coverage percentage (0-100)
    """
    if mask is None or mask.size == 0:
        return 0.0

    total_pixels = mask.size
    masked_pixels = np.count_nonzero(mask)

    return (masked_pixels / total_pixels) * 100


def smooth_mask_edges(
    mask: np.ndarray,
    blur_size: int = 5
) -> np.ndarray:
    """
    Apply Gaussian blur to soften mask edges.

    Helps reduce visible seams after inpainting.

    Args:
        mask: Binary mask
        blur_size: Gaussian kernel size (must be odd)

    Returns:
        Smoothed mask
    """
    if blur_size % 2 == 0:
        blur_size += 1

    # Normalize to 0-255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    smoothed = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return smoothed


def validate_mask(
    mask: np.ndarray,
    frame_shape: Tuple[int, int, int],
    max_coverage_pct: float = 15.0
) -> dict:
    """
    Validate mask quality and coverage.

    Args:
        mask: Binary mask to validate
        frame_shape: Expected (height, width, channels)
        max_coverage_pct: Maximum allowed coverage percentage

    Returns:
        Dict with validation results
    """
    height, width = frame_shape[:2]

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "coverage_pct": 0.0
    }

    # Check mask exists
    if mask is None:
        results["valid"] = False
        results["errors"].append("Mask is None")
        return results

    # Check dimensions match
    if mask.shape != (height, width):
        results["valid"] = False
        results["errors"].append(
            f"Mask shape {mask.shape} doesn't match frame {(height, width)}"
        )
        return results

    # Check coverage
    coverage = get_mask_coverage(mask)
    results["coverage_pct"] = coverage

    if coverage == 0:
        results["warnings"].append("Mask is empty (no watermark detected)")
    elif coverage > max_coverage_pct:
        results["warnings"].append(
            f"Mask covers {coverage:.1f}% of frame (expected <{max_coverage_pct}%)"
        )

    return results
