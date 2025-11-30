#!/usr/bin/env python3
"""
SAM3 Segmentation Wrapper for Precise Watermark Masking

Uses Segment Anything Model (SAM2/SAM3) to generate precise
segmentation masks from YOLO-World bounding boxes.

Captures:
- Text pixels
- Glow/animation effects
- Full watermark region including @username variations
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging

from .mask_utils import expand_bbox, dilate_mask, merge_masks, create_empty_mask

# Fix Windows console encoding for SAM3's Unicode output (checkmarks, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Ignore if reconfigure not available

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """
    SAM2/SAM3 wrapper for precise watermark segmentation.

    Takes bounding boxes from YOLO-World and generates
    pixel-precise masks that capture the full watermark
    including glow and animation effects.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize SAM3 segmenter.

        Args:
            model_path: Path to SAM checkpoint (.pt file)
            model_cfg: SAM model configuration name
            device: Device to run on ("cuda" or "cpu")
            use_fp16: Use fp16 for memory efficiency
        """
        self.model_path = Path(model_path)
        self.model_cfg = model_cfg
        self.device = device
        self.use_fp16 = use_fp16

        self.predictor = None
        self._current_image = None
        self._load_model()

    def _load_model(self):
        """Load SAM2/SAM3 model."""
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading SAM from {self.model_path}")

            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.use_fp16 = False

            # Load predictor
            self.predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large"
            )

            # Alternative: load from local checkpoint
            # from sam2.build_sam import build_sam2
            # sam = build_sam2(self.model_cfg, str(self.model_path))
            # self.predictor = SAM2ImagePredictor(sam)

            logger.info(f"SAM loaded on {self.device}")

        except ImportError as e:
            logger.error(f"SAM2 import failed: {e}")
            raise ImportError(
                "SAM2 is required. Install: pip install git+https://github.com/facebookresearch/sam2.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {e}")

    def set_image(self, image: np.ndarray):
        """
        Set image for segmentation (computes embeddings once).

        Args:
            image: BGR or RGB image as numpy array (H, W, C)
        """
        import torch

        self._current_image = image

        # SAM expects RGB
        if image.shape[-1] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image

        with torch.inference_mode():
            if self.use_fp16 and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.predictor.set_image(image_rgb)
            else:
                self.predictor.set_image(image_rgb)

    def segment_from_bbox(
        self,
        bbox: List[float],
        expand_px: int = 10,
        multimask_output: bool = False,
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Segment region from bounding box.

        Expands bbox to capture glow effects, then uses SAM
        to generate precise segmentation mask.

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
            expand_px: Pixels to expand bbox before segmentation
            multimask_output: Return multiple mask proposals
            return_scores: Also return IoU confidence scores

        Returns:
            Binary mask (H, W) as uint8 (0-255), optionally with scores
        """
        import torch

        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image() first.")

        if self.predictor is None:
            raise RuntimeError("Model not loaded")

        # Expand bbox to capture glow/effects
        expanded_bbox = expand_bbox(
            bbox,
            expand_px,
            self._current_image.shape
        )

        # Convert to numpy array for SAM
        box = np.array(expanded_bbox, dtype=np.float32)

        with torch.inference_mode():
            if self.use_fp16 and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :],  # Add batch dimension
                        multimask_output=multimask_output
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=multimask_output
                )

        # Get best mask (highest IoU score)
        if multimask_output:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0]

        # Convert to uint8 (0-255)
        mask = (mask.astype(np.uint8) * 255)

        if return_scores:
            return mask, float(score)
        return mask

    def segment_from_bboxes(
        self,
        bboxes: List[List[float]],
        expand_px: int = 10,
        merge_output: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Segment multiple regions from bounding boxes.

        Args:
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            expand_px: Pixels to expand each bbox
            merge_output: Merge all masks into single mask

        Returns:
            Merged mask or list of individual masks
        """
        if not bboxes:
            if self._current_image is not None:
                return create_empty_mask(self._current_image.shape)
            raise ValueError("No bboxes provided and no image set")

        masks = []
        for bbox in bboxes:
            mask = self.segment_from_bbox(bbox, expand_px=expand_px)
            masks.append(mask)

        if merge_output:
            return merge_masks(masks)

        return masks

    def segment_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        expand_px: int = 10,
        dilate_px: int = 3
    ) -> np.ndarray:
        """
        Full pipeline: segment all detections in a frame.

        Args:
            frame: BGR image as numpy array
            detections: List of detection dicts from YOLOWorldDetector
            expand_px: Pixels to expand bboxes
            dilate_px: Pixels to dilate final mask

        Returns:
            Final merged and dilated mask
        """
        if not detections:
            return create_empty_mask(frame.shape)

        # Set image (computes embeddings)
        self.set_image(frame)

        # Extract bboxes from detections
        bboxes = [det["bbox"] for det in detections]

        # Segment all regions
        merged_mask = self.segment_from_bboxes(
            bboxes,
            expand_px=expand_px,
            merge_output=True
        )

        # Apply dilation for complete coverage
        if dilate_px > 0:
            merged_mask = dilate_mask(merged_mask, kernel_size=dilate_px * 2 + 1)

        return merged_mask

    def segment_with_point_refinement(
        self,
        bbox: List[float],
        positive_points: Optional[List[Tuple[int, int]]] = None,
        negative_points: Optional[List[Tuple[int, int]]] = None,
        expand_px: int = 10
    ) -> np.ndarray:
        """
        Segment with optional point prompts for refinement.

        Useful when bbox alone doesn't capture the full watermark.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            positive_points: Points to include in mask [(x, y), ...]
            negative_points: Points to exclude from mask [(x, y), ...]
            expand_px: Pixels to expand bbox

        Returns:
            Refined segmentation mask
        """
        import torch

        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image() first.")

        # Expand bbox
        expanded_bbox = expand_bbox(
            bbox,
            expand_px,
            self._current_image.shape
        )
        box = np.array(expanded_bbox, dtype=np.float32)

        # Prepare point prompts
        point_coords = None
        point_labels = None

        if positive_points or negative_points:
            all_points = []
            all_labels = []

            if positive_points:
                all_points.extend(positive_points)
                all_labels.extend([1] * len(positive_points))

            if negative_points:
                all_points.extend(negative_points)
                all_labels.extend([0] * len(negative_points))

            point_coords = np.array(all_points, dtype=np.float32)
            point_labels = np.array(all_labels, dtype=np.int32)

        with torch.inference_mode():
            if self.use_fp16 and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box[None, :],
                        multimask_output=True
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box[None, :],
                    multimask_output=True
                )

        # Get best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        return (mask.astype(np.uint8) * 255)

    def reset(self):
        """Reset internal state (clear cached image)."""
        self._current_image = None

    def __repr__(self):
        return (
            f"SAM3Segmenter("
            f"model={self.model_path.name}, "
            f"device={self.device}, "
            f"fp16={self.use_fp16})"
        )


class SAM3SegmenterLocal:
    """
    Alternative SAM3 segmenter that loads from local checkpoint.

    Use this if you have a custom SAM3.pt file rather than
    using the HuggingFace pretrained model.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Initialize SAM3 from local checkpoint.

        Args:
            checkpoint_path: Path to SAM3.pt checkpoint
            model_cfg: Model configuration file name
            device: Device to run on
            use_fp16: Use fp16 inference
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_cfg = model_cfg
        self.device = device
        self.use_fp16 = use_fp16

        self.predictor = None
        self._current_image = None
        self._load_model()

    def _load_model(self):
        """Load SAM from local checkpoint."""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            logger.info(f"Loading SAM from local checkpoint: {self.checkpoint_path}")

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.use_fp16 = False

            # Build SAM from checkpoint
            sam = build_sam2(
                self.model_cfg,
                str(self.checkpoint_path),
                device=self.device
            )

            self.predictor = SAM2ImagePredictor(sam)
            logger.info(f"SAM loaded from {self.checkpoint_path.name}")

        except ImportError as e:
            raise ImportError(f"SAM2 import failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM checkpoint: {e}")

    # Inherit methods from SAM3Segmenter
    set_image = SAM3Segmenter.set_image
    segment_from_bbox = SAM3Segmenter.segment_from_bbox
    segment_from_bboxes = SAM3Segmenter.segment_from_bboxes
    segment_frame = SAM3Segmenter.segment_frame
    segment_with_point_refinement = SAM3Segmenter.segment_with_point_refinement
    reset = SAM3Segmenter.reset


class SAM3TextSegmenter:
    """
    Meta SAM3 (Segment Anything Model 3) with text prompt support.

    This uses the actual SAM3 model from Meta which supports:
    - Text prompts: "SORA", "watermark", "text" etc.
    - No need for separate detection (YOLO) - SAM3 detects AND segments
    - Better accuracy for open-vocabulary concepts

    Requires Python 3.12+ and the sam3 library.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        bpe_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        use_fp16: bool = True
    ):
        """
        Initialize SAM3 text segmenter.

        Args:
            checkpoint_path: Path to sam3.pt checkpoint
            bpe_path: Path to BPE vocab file (optional, uses default)
            device: Device to run on ("cuda" or "cpu")
            confidence_threshold: Detection confidence threshold
            use_fp16: Use bfloat16 for efficiency
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.bpe_path = bpe_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16

        self.model = None
        self.processor = None
        self._current_image = None
        self._inference_state = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 model."""
        try:
            import torch
            import sam3
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info(f"Loading SAM3 from {self.checkpoint_path}")

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.use_fp16 = False

            # Enable tf32 for Ampere GPUs
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Get BPE path from sam3 package if not provided
            if self.bpe_path is None:
                import os
                sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
                self.bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

            # Build model
            self.model = build_sam3_image_model(
                bpe_path=str(self.bpe_path),
                checkpoint_path=str(self.checkpoint_path),
                load_from_HF=False,
                device=self.device
            )

            # Create processor
            self.processor = Sam3Processor(
                self.model,
                confidence_threshold=self.confidence_threshold
            )

            logger.info(f"SAM3 loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"SAM3 import failed: {e}. "
                "SAM3 requires Python 3.12+. Install: pip install git+https://github.com/facebookresearch/sam3.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def set_image(self, image: np.ndarray):
        """
        Set image for segmentation.

        Args:
            image: BGR or RGB image as numpy array (H, W, C)
        """
        from PIL import Image as PILImage

        self._current_image = image

        # Convert BGR to RGB if needed
        if image.shape[-1] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image

        # Convert to PIL Image
        pil_image = PILImage.fromarray(image_rgb)

        # Set image and get inference state
        self._inference_state = self.processor.set_image(pil_image)

    def segment_from_text(
        self,
        prompts: Union[str, List[str]],
        dilate_px: int = 3
    ) -> np.ndarray:
        """
        Segment regions matching text prompts.

        This is the key SAM3 feature - text-based detection AND segmentation
        in one step. No need for YOLO-World!

        Args:
            prompts: Text prompt(s) like "SORA", "watermark", "text"
            dilate_px: Pixels to dilate final mask

        Returns:
            Binary mask (H, W) as uint8 (0-255)
        """
        if self._current_image is None:
            raise RuntimeError("No image set. Call set_image() first.")

        if isinstance(prompts, str):
            prompts = [prompts]

        all_masks = []

        for prompt in prompts:
            # Reset prompts and set new text prompt
            self.processor.reset_all_prompts(self._inference_state)
            self._inference_state = self.processor.set_text_prompt(
                state=self._inference_state,
                prompt=prompt
            )

            # Extract masks from inference state (state is a dict)
            masks = self._inference_state.get('masks')
            if masks is not None and masks.shape[0] > 0:
                # masks shape is [N, 1, H, W], squeeze the channel dim
                masks_np = masks.squeeze(1).cpu().numpy()  # [N, H, W]
                for mask_np in masks_np:
                    all_masks.append(mask_np)

        if not all_masks:
            return create_empty_mask(self._current_image.shape)

        # Merge all masks
        merged = merge_masks([(m * 255).astype(np.uint8) for m in all_masks])

        # Apply dilation
        if dilate_px > 0:
            merged = dilate_mask(merged, kernel_size=dilate_px * 2 + 1)

        return merged

    def segment_frame_with_text(
        self,
        frame: np.ndarray,
        prompts: List[str] = None,
        dilate_px: int = 3
    ) -> np.ndarray:
        """
        Full pipeline: detect and segment watermarks using text prompts.

        Args:
            frame: BGR image as numpy array
            prompts: Text prompts (default: ["SORA", "watermark", "text", "logo"])
            dilate_px: Pixels to dilate final mask

        Returns:
            Final merged and dilated mask
        """
        if prompts is None:
            prompts = ["SORA", "watermark", "text", "logo", "username"]

        self.set_image(frame)
        return self.segment_from_text(prompts, dilate_px=dilate_px)

    def reset(self):
        """Reset internal state."""
        self._current_image = None
        self._inference_state = None

    def __repr__(self):
        return (
            f"SAM3TextSegmenter("
            f"checkpoint={self.checkpoint_path.name}, "
            f"device={self.device}, "
            f"threshold={self.confidence_threshold})"
        )
