#!/usr/bin/env python3
"""
Standalone Watermark Removal Pipeline

End-to-end video watermark removal WITHOUT ComfyUI dependency.
Uses: YOLO-World → SAM3 → ProPainter → FFmpeg

Pipeline:
    Input Video → Extract Frames → Detect (YOLO-World) →
    Segment (SAM3) → Inpaint (ProPainter) → Stitch + Audio → Output
"""

import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
import logging
import json

import numpy as np
import cv2

from ..core.audio_preservation import AudioPreservationPipeline
from ..core.yolo_detector import YOLOWorldDetector
from ..core.sam3_segmenter import SAM3Segmenter, SAM3SegmenterLocal, SAM3TextSegmenter
from ..core.sam3_multigpu import SAM3TextSegmenterMultiGPU
from ..core.propainter import ProPainterInpainter, ProPainterInpainterLite
from ..core.mask_utils import (
    dilate_mask, merge_masks, create_empty_mask,
    save_mask, validate_mask, get_mask_coverage
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the standalone watermark removal pipeline."""

    # Model paths
    yolo_model_path: str = "models/yolov8x-worldv2.pt"
    sam_model_path: str = "models/sam2_hiera_large.pt"
    sam_model_cfg: str = "sam2_hiera_l.yaml"
    propainter_dir: str = "vendor/ProPainter"
    propainter_weights_dir: Optional[str] = None

    # Detection settings
    detection_classes: List[str] = None  # None = use defaults
    detection_conf_threshold: float = 0.15
    detection_iou_threshold: float = 0.45

    # Segmentation settings
    bbox_expand_px: int = 10  # Expand bbox to capture glow
    mask_dilate_px: int = 5   # Dilate mask for complete coverage

    # Inpainting settings
    neighbor_length: int = 10
    ref_stride: int = 10
    subvideo_length: int = 40  # Reduced from 80 to prevent OOM on HD video
    resize_ratio: float = 0.5  # Scale down during ProPainter processing for memory

    # Processing settings
    device: str = "cuda"
    use_fp16: bool = True
    use_sam_local: bool = True  # Use local SAM checkpoint vs HuggingFace
    use_sam3_text: bool = False  # Use SAM3 text prompts (requires Python 3.12+, skips YOLO)
    use_sam3_multigpu: bool = False  # Split SAM3 across multiple GPUs
    vision_device: str = "cuda:0"  # GPU for vision backbone (larger VRAM)
    language_device: str = "cuda:1"  # GPU for language backbone (smaller VRAM)

    # Output settings
    output_crf: int = 18  # Video quality (18=high, 23=default)
    output_preset: str = "medium"
    cleanup_temp: bool = True

    def __post_init__(self):
        if self.detection_classes is None:
            self.detection_classes = [
                "Sora", "text overlay", "white text", "watermark", "username", "@username"
            ]


@dataclass
class PipelineResult:
    """Result from the watermark removal pipeline."""

    success: bool
    output_path: Optional[Path] = None
    input_path: Optional[Path] = None

    # Statistics
    total_frames: int = 0
    frames_with_watermark: int = 0
    processing_time_seconds: float = 0.0
    average_mask_coverage_pct: float = 0.0

    # Audio validation
    audio_synced: bool = False
    duration_diff_ms: float = 0.0

    # Errors/warnings
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "input_path": str(self.input_path) if self.input_path else None,
            "total_frames": self.total_frames,
            "frames_with_watermark": self.frames_with_watermark,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "average_mask_coverage_pct": round(self.average_mask_coverage_pct, 2),
            "audio_synced": self.audio_synced,
            "duration_diff_ms": round(self.duration_diff_ms, 2),
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class StandaloneWatermarkJob:
    """
    End-to-end watermark removal pipeline (no ComfyUI dependency).

    Orchestrates:
    1. Frame extraction (FFmpeg via AudioPreservationPipeline)
    2. Watermark detection (YOLO-World)
    3. Precise segmentation (SAM3)
    4. Video inpainting (ProPainter)
    5. Video stitching with audio preservation

    Usage:
        config = PipelineConfig(
            yolo_model_path="models/yolo-world.pt",
            sam_model_path="models/SAM3.pt",
            propainter_dir="vendor/ProPainter"
        )
        job = StandaloneWatermarkJob(config)
        result = job.run("input.mp4", "output.mp4")
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: PipelineConfig with model paths and settings
        """
        self.config = config

        # Initialize components (lazy loaded)
        self._audio_pipeline = None
        self._detector = None
        self._segmenter = None
        self._inpainter = None

        # Temp directories
        self._temp_dir = None

    @property
    def audio_pipeline(self) -> AudioPreservationPipeline:
        """Lazy load audio pipeline."""
        if self._audio_pipeline is None:
            self._audio_pipeline = AudioPreservationPipeline()
        return self._audio_pipeline

    @property
    def detector(self) -> YOLOWorldDetector:
        """Lazy load YOLO-World detector."""
        if self._detector is None:
            logger.info(f"Loading YOLO-World from {self.config.yolo_model_path}")
            self._detector = YOLOWorldDetector(
                model_path=self.config.yolo_model_path,
                classes=self.config.detection_classes,
                device=self.config.device,
                conf_threshold=self.config.detection_conf_threshold,
                iou_threshold=self.config.detection_iou_threshold
            )
        return self._detector

    @property
    def segmenter(self) -> Union[SAM3Segmenter, SAM3SegmenterLocal, SAM3TextSegmenter]:
        """Lazy load SAM3 segmenter."""
        if self._segmenter is None:
            logger.info(f"Loading SAM3 from {self.config.sam_model_path}")
            if self.config.use_sam3_text:
                # Use SAM3 with text prompts (requires Python 3.12+)
                if self.config.use_sam3_multigpu:
                    # Multi-GPU: vision on one GPU, language on another
                    logger.info(f"Using multi-GPU SAM3: vision={self.config.vision_device}, language={self.config.language_device}")
                    self._segmenter = SAM3TextSegmenterMultiGPU(
                        checkpoint_path=self.config.sam_model_path,
                        vision_device=self.config.vision_device,
                        language_device=self.config.language_device,
                        confidence_threshold=self.config.detection_conf_threshold,
                        use_fp16=self.config.use_fp16
                    )
                else:
                    self._segmenter = SAM3TextSegmenter(
                        checkpoint_path=self.config.sam_model_path,
                        device=self.config.device,
                        confidence_threshold=self.config.detection_conf_threshold,
                        use_fp16=self.config.use_fp16
                    )
            elif self.config.use_sam_local:
                self._segmenter = SAM3SegmenterLocal(
                    checkpoint_path=self.config.sam_model_path,
                    model_cfg=self.config.sam_model_cfg,
                    device=self.config.device,
                    use_fp16=self.config.use_fp16
                )
            else:
                self._segmenter = SAM3Segmenter(
                    model_path=self.config.sam_model_path,
                    model_cfg=self.config.sam_model_cfg,
                    device=self.config.device,
                    use_fp16=self.config.use_fp16
                )
        return self._segmenter

    @property
    def inpainter(self) -> ProPainterInpainter:
        """Lazy load ProPainter inpainter."""
        if self._inpainter is None:
            logger.info(f"Loading ProPainter from {self.config.propainter_dir}")
            self._inpainter = ProPainterInpainter(
                propainter_dir=self.config.propainter_dir,
                weights_dir=self.config.propainter_weights_dir,
                device=self.config.device,
                use_fp16=self.config.use_fp16
            )
        return self._inpainter

    def run(
        self,
        input_video: Union[str, Path],
        output_video: Union[str, Path],
        temp_dir: Optional[Union[str, Path]] = None
    ) -> PipelineResult:
        """
        Run the full watermark removal pipeline.

        Args:
            input_video: Path to input video
            output_video: Path to save output video
            temp_dir: Optional temp directory (auto-cleaned if config.cleanup_temp)

        Returns:
            PipelineResult with success status and statistics
        """
        start_time = time.time()

        input_video = Path(input_video)
        output_video = Path(output_video)

        if temp_dir:
            self._temp_dir = Path(temp_dir)
        else:
            self._temp_dir = Path("temp") / f"job_{input_video.stem}_{int(time.time())}"

        self._temp_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            success=False,
            input_path=input_video,
            output_path=output_video
        )

        try:
            # Validate input
            if not input_video.exists():
                raise FileNotFoundError(f"Input video not found: {input_video}")

            logger.info(f"Starting watermark removal: {input_video}")

            # Step 1: Extract frames
            logger.info("Step 1/5: Extracting frames...")
            frames_dir = self._extract_frames(input_video)
            frame_files = sorted(frames_dir.glob("*.png"))
            result.total_frames = len(frame_files)
            logger.info(f"Extracted {result.total_frames} frames")

            # Get original frame dimensions for upscaling after ProPainter
            original_size = None
            if frame_files:
                sample = cv2.imread(str(frame_files[0]))
                if sample is not None:
                    h, w = sample.shape[:2]
                    original_size = (w, h)  # (width, height) for ffmpeg

            # Step 2: Detect and segment watermarks
            logger.info("Step 2/5: Detecting and segmenting watermarks...")
            masks_dir = self._temp_dir / "masks"
            masks_dir.mkdir(exist_ok=True)

            detection_stats = self._detect_and_segment(frames_dir, masks_dir)
            result.frames_with_watermark = detection_stats["frames_with_detection"]
            result.average_mask_coverage_pct = detection_stats["avg_coverage"]

            if result.frames_with_watermark == 0:
                result.warnings.append("No watermarks detected in any frame")
                logger.warning("No watermarks detected - copying original video")
                shutil.copy(input_video, output_video)
                result.success = True
                result.processing_time_seconds = time.time() - start_time
                return result

            logger.info(
                f"Detected watermarks in {result.frames_with_watermark}/{result.total_frames} frames "
                f"(avg coverage: {result.average_mask_coverage_pct:.1f}%)"
            )

            # Step 3: Inpaint with ProPainter
            logger.info("Step 3/5: Inpainting with ProPainter...")
            inpainted_dir = self._temp_dir / "inpainted"
            self._inpaint_frames(frames_dir, masks_dir, inpainted_dir)

            # Step 4: Stitch video with audio
            logger.info("Step 4/5: Stitching video with audio...")
            self._stitch_video(inpainted_dir, input_video, output_video, original_size)

            # Step 5: Validate output
            logger.info("Step 5/5: Validating output...")
            validation = self.audio_pipeline.validate_audio_sync(
                str(input_video), str(output_video)
            )
            result.audio_synced = validation["synced"]
            result.duration_diff_ms = validation["duration_diff_ms"]

            if not result.audio_synced:
                result.warnings.append(
                    f"Audio sync issue: {result.duration_diff_ms:.2f}ms difference"
                )

            result.success = True
            result.processing_time_seconds = time.time() - start_time

            logger.info(
                f"Watermark removal complete in {result.processing_time_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.error_message = str(e)
            result.processing_time_seconds = time.time() - start_time

        finally:
            # Cleanup temp files
            if self.config.cleanup_temp and self._temp_dir and self._temp_dir.exists():
                logger.info("Cleaning up temporary files...")
                shutil.rmtree(self._temp_dir, ignore_errors=True)

        return result

    def _extract_frames(self, input_video: Path) -> Path:
        """Extract frames from video."""
        frames_dir = self._temp_dir / "frames"
        return self.audio_pipeline.extract_frames(
            str(input_video),
            output_dir=str(frames_dir)
        )

    def _detect_and_segment(
        self,
        frames_dir: Path,
        masks_dir: Path
    ) -> Dict:
        """
        Detect watermarks and generate segmentation masks for all frames.

        Includes backward propagation: copies mask to N-3 frames before first detection
        to catch watermarks that fade in gradually.

        Returns statistics about detection coverage.
        """
        frame_files = sorted(frames_dir.glob("*.png"))
        frames_with_detection = 0
        total_coverage = 0.0

        # Track first detection for backward propagation
        first_detection_frame = None
        first_detection_mask = None

        for i, frame_path in enumerate(frame_files):
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                # Save empty mask
                empty_mask = create_empty_mask(frame.shape if frame is not None else (720, 1280, 3))
                save_mask(empty_mask, masks_dir / frame_path.name)
                continue

            # Detect and segment watermarks
            if self.config.use_sam3_text:
                # SAM3 text mode: detect AND segment in one step
                mask = self.segmenter.segment_frame_with_text(
                    frame,
                    prompts=self.config.detection_classes,
                    dilate_px=self.config.mask_dilate_px
                )
                
                if np.any(mask > 0):
                    frames_with_detection += 1
                    coverage = get_mask_coverage(mask)
                    total_coverage += coverage

                    # Track first detection for backward propagation
                    if first_detection_frame is None:
                        first_detection_frame = i
                        first_detection_mask = mask.copy()
                        logger.debug(f"First watermark detection at frame {i}")

                    validation = validate_mask(mask, frame.shape)
                    if validation["warnings"]:
                        for w in validation["warnings"]:
                            logger.debug(f"Frame {i}: {w}")
            else:
                # Standard mode: YOLO detect then SAM segment
                detections = self.detector.detect(frame)

                if detections:
                    # Merge overlapping detections
                    detections = self.detector.merge_overlapping_detections(
                        detections, iou_threshold=0.3
                    )

                    # Segment with SAM
                    mask = self.segmenter.segment_frame(
                        frame,
                        detections,
                        expand_px=self.config.bbox_expand_px,
                        dilate_px=self.config.mask_dilate_px
                    )

                    frames_with_detection += 1
                    coverage = get_mask_coverage(mask)
                    total_coverage += coverage

                    # Track first detection for backward propagation
                    if first_detection_frame is None:
                        first_detection_frame = i
                        first_detection_mask = mask.copy()
                        logger.debug(f"First watermark detection at frame {i}")

                    # Validate mask
                    validation = validate_mask(mask, frame.shape)
                    if validation["warnings"]:
                        for w in validation["warnings"]:
                            logger.debug(f"Frame {i}: {w}")

                else:
                    # No detection - create empty mask
                    mask = create_empty_mask(frame.shape)

            # Save mask
            save_mask(mask, masks_dir / frame_path.name)

            # Progress logging
            if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                logger.info(f"Processed {i + 1}/{len(frame_files)} frames")

        # Reset segmenter state
        self.segmenter.reset()

        # Backward propagation: copy first detection mask to N-3 preceding frames
        # This catches watermarks that fade in gradually
        backward_frames = 3
        if first_detection_frame is not None and first_detection_frame > 0:
            start_frame = max(0, first_detection_frame - backward_frames)
            logger.info(
                f"Backward propagation: copying mask from frame {first_detection_frame} "
                f"to frames {start_frame}-{first_detection_frame - 1}"
            )
            for j in range(start_frame, first_detection_frame):
                frame_name = frame_files[j].name
                save_mask(first_detection_mask, masks_dir / frame_name)
                frames_with_detection += 1
                total_coverage += get_mask_coverage(first_detection_mask)

        avg_coverage = (total_coverage / frames_with_detection) if frames_with_detection > 0 else 0.0

        return {
            "frames_with_detection": frames_with_detection,
            "avg_coverage": avg_coverage
        }

    def _inpaint_frames(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path
    ) -> Path:
        """Inpaint frames using ProPainter."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get frame dimensions for ProPainter
        frame_files = list(frames_dir.glob("*.png"))
        if frame_files:
            sample_frame = cv2.imread(str(frame_files[0]))
            height, width = sample_frame.shape[:2]
        else:
            height, width = None, None

        # Run ProPainter
        result_dir = self.inpainter.inpaint_video(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            height=height,
            width=width,
            neighbor_length=self.config.neighbor_length,
            ref_stride=self.config.ref_stride,
            subvideo_length=self.config.subvideo_length,
            resize_ratio=self.config.resize_ratio
        )

        # Rename output frames to match expected pattern if needed
        self._normalize_frame_names(result_dir, output_dir)

        return output_dir

    def _normalize_frame_names(self, source_dir: Path, target_dir: Path):
        """Ensure output frames follow %06d.png naming convention."""
        frame_files = sorted(
            list(source_dir.glob("*.png")) +
            list(source_dir.glob("*.jpg"))
        )

        for i, src_file in enumerate(frame_files):
            target_name = f"{i + 1:06d}.png"
            target_path = target_dir / target_name

            if src_file != target_path:
                # Read and save as PNG
                frame = cv2.imread(str(src_file))
                if frame is not None:
                    cv2.imwrite(str(target_path), frame)

    def _stitch_video(
        self,
        frames_dir: Path,
        original_video: Path,
        output_video: Path,
        original_size: tuple = None
    ):
        """Stitch frames back to video with audio."""
        # If resize_ratio was used, we need to scale back to original size
        target_size = None
        if self.config.resize_ratio != 1.0 and original_size:
            target_size = original_size  # (width, height)

        self.audio_pipeline.stitch_with_audio(
            frames_dir=str(frames_dir),
            original_video=str(original_video),
            output_video=str(output_video),
            crf=self.config.output_crf,
            preset=self.config.output_preset,
            target_size=target_size
        )

    def process_single_frame(
        self,
        frame: np.ndarray,
        return_mask: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Process a single frame (for testing/preview).

        Note: For video, use run() for temporal consistency.

        Args:
            frame: BGR image as numpy array
            return_mask: Also return the generated mask

        Returns:
            Inpainted frame, optionally with mask
        """
        # Detect
        detections = self.detector.detect(frame)

        if not detections:
            if return_mask:
                return frame, create_empty_mask(frame.shape)
            return frame

        # Segment
        mask = self.segmenter.segment_frame(
            frame,
            detections,
            expand_px=self.config.bbox_expand_px,
            dilate_px=self.config.mask_dilate_px
        )

        # Inpaint (use lite fallback for single frame)
        lite_inpainter = ProPainterInpainterLite(device=self.config.device)
        result = lite_inpainter.inpaint_frame(frame, mask)

        if return_mask:
            return result, mask
        return result

    def __repr__(self):
        return (
            f"StandaloneWatermarkJob("
            f"yolo={Path(self.config.yolo_model_path).name}, "
            f"sam={Path(self.config.sam_model_path).name}, "
            f"device={self.config.device})"
        )


def run_pipeline(
    input_video: str,
    output_video: str,
    yolo_model: str = "models/yolo-world.pt",
    sam_model: str = "models/SAM3.pt",
    propainter_dir: str = "vendor/ProPainter",
    device: str = "cuda",
    use_fp16: bool = True
) -> PipelineResult:
    """
    Convenience function to run the watermark removal pipeline.

    Args:
        input_video: Path to input video
        output_video: Path for output video
        yolo_model: Path to YOLO-World weights
        sam_model: Path to SAM3 checkpoint
        propainter_dir: Path to ProPainter repository
        device: Device to run on
        use_fp16: Use half precision

    Returns:
        PipelineResult with success status
    """
    config = PipelineConfig(
        yolo_model_path=yolo_model,
        sam_model_path=sam_model,
        propainter_dir=propainter_dir,
        device=device,
        use_fp16=use_fp16
    )

    job = StandaloneWatermarkJob(config)
    return job.run(input_video, output_video)
