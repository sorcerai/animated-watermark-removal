#!/usr/bin/env python3
"""
Animated Watermark Remover - CLI Entry Point

Removes animated watermarks (SORA + @username) from videos using:
- YOLO-World for open-vocabulary detection
- SAM3 for precise segmentation
- ProPainter for temporal-consistent inpainting

Usage:
    python remove_text.py --input video.mp4 --output clean.mp4

    # With custom models
    python remove_text.py --input video.mp4 --output clean.mp4 \
        --yolo-model models/yolo-world-l.pt \
        --sam-model models/SAM3.pt \
        --propainter-dir vendor/ProPainter

    # CPU mode (slower)
    python remove_text.py --input video.mp4 --output clean.mp4 --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.standalone import StandaloneWatermarkJob, PipelineConfig, PipelineResult


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove Sora watermarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python remove_text.py --input sora_video.mp4 --output clean_video.mp4

  # Custom detection targets
  python remove_text.py --input video.mp4 --output clean.mp4 \
    --target-text "SORA,text,watermark,logo"

  # Memory-efficient mode (for long videos)
  python remove_text.py --input long_video.mp4 --output clean.mp4 \
    --subvideo-length 40 --neighbor-length 5 --ref-stride 15

  # Preview single frame (for testing)
  python remove_text.py --input video.mp4 --preview-frame 100
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=False,
        help="Output video file path (default: input_clean.mp4)"
    )

    # Model paths
    parser.add_argument(
        "--yolo-model",
        default="models/yolov8x-worldv2.pt",
        help="Path to YOLO-World weights (default: models/yolov8x-worldv2.pt)"
    )
    parser.add_argument(
        "--sam-model",
        default="models/sam3.pt",
        help="Path to SAM3 checkpoint (default: models/sam3.pt)"
    )
    parser.add_argument(
        "--sam-config",
        default="sam2_hiera_l.yaml",
        help="SAM model config name (default: sam2_hiera_l.yaml)"
    )
    parser.add_argument(
        "--propainter-dir",
        default="vendor/ProPainter",
        help="Path to ProPainter repository (default: vendor/ProPainter)"
    )
    parser.add_argument(
        "--propainter-weights",
        default=None,
        help="Path to ProPainter weights directory (default: propainter_dir/weights)"
    )

    # Detection settings
    parser.add_argument(
        "--target-text",
        default="word,text,watermark,letters",
        help="Comma-separated text prompts for SAM3 (default: word,text,watermark,letters)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15 for SAM3)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Detection NMS IoU threshold (default: 0.45)"
    )

    # Segmentation settings
    parser.add_argument(
        "--bbox-expand",
        type=int,
        default=10,
        help="Pixels to expand bbox for glow capture (default: 10)"
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=3,
        help="Pixels to dilate mask for complete coverage (default: 3)"
    )

    # Inpainting settings
    parser.add_argument(
        "--neighbor-length",
        type=int,
        default=10,
        help="ProPainter local neighbor frames (default: 10)"
    )
    parser.add_argument(
        "--ref-stride",
        type=int,
        default=10,
        help="ProPainter global reference stride (default: 10)"
    )
    parser.add_argument(
        "--subvideo-length",
        type=int,
        default=40,
        help="Frames per sub-video chunk for memory (default: 40, reduced for HD video)"
    )
    parser.add_argument(
        "--resize-ratio",
        type=float,
        default=0.5,
        help="Scale factor for ProPainter processing (default: 0.5 for HD video)"
    )

    # Processing settings
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 (half precision) inference"
    )
    parser.add_argument(
        "--no-sam3-text",
        action="store_true",
        help="Disable SAM3 text mode, use YOLO + SAM2 instead"
    )
    parser.add_argument(
        "--use-huggingface-sam",
        action="store_true",
        help="Use HuggingFace SAM instead of local checkpoint (only with --no-sam3-text)"
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
        help="Split SAM3 across multiple GPUs (vision on --vision-device, language on --language-device)"
    )
    parser.add_argument(
        "--vision-device",
        default="cuda:0",
        help="GPU for vision backbone (default: cuda:0, should be larger VRAM)"
    )
    parser.add_argument(
        "--language-device",
        default="cuda:1",
        help="GPU for language backbone (default: cuda:1, can be smaller VRAM)"
    )

    # Output settings
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Output video quality CRF (0-51, lower=better, default: 18)"
    )
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"],
        default="medium",
        help="Encoding preset (default: medium)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing"
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Custom temporary directory path"
    )

    # Preview/debug
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=None,
        help="Process and save single frame for preview (frame number)"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save detection masks to output directory"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )

    return parser.parse_args()


def preview_single_frame(args):
    """Process and save a single frame for preview."""
    import cv2
    import numpy as np

    input_path = Path(args.input)
    frame_num = args.preview_frame

    # Extract frame from video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return False

    # Build config
    config = build_config(args)
    job = StandaloneWatermarkJob(config)

    # Process single frame
    print(f"Processing frame {frame_num}...")
    result, mask = job.process_single_frame(frame, return_mask=True)

    # Save results
    output_dir = input_path.parent / f"{input_path.stem}_preview"
    output_dir.mkdir(exist_ok=True)

    frame_path = output_dir / f"frame_{frame_num:06d}_original.png"
    mask_path = output_dir / f"frame_{frame_num:06d}_mask.png"
    result_path = output_dir / f"frame_{frame_num:06d}_result.png"

    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(result_path), result)

    # Create comparison image
    comparison = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
    comparison_path = output_dir / f"frame_{frame_num:06d}_comparison.png"
    cv2.imwrite(str(comparison_path), comparison)

    print(f"Preview saved to: {output_dir}")
    print(f"  - Original: {frame_path.name}")
    print(f"  - Mask: {mask_path.name}")
    print(f"  - Result: {result_path.name}")
    print(f"  - Comparison: {comparison_path.name}")

    return True


def build_config(args) -> PipelineConfig:
    """Build pipeline configuration from arguments."""
    return PipelineConfig(
        yolo_model_path=args.yolo_model,
        sam_model_path=args.sam_model,
        sam_model_cfg=args.sam_config,
        propainter_dir=args.propainter_dir,
        propainter_weights_dir=args.propainter_weights,
        detection_classes=[t.strip() for t in args.target_text.split(",")],
        detection_conf_threshold=args.conf_threshold,
        detection_iou_threshold=args.iou_threshold,
        bbox_expand_px=args.bbox_expand,
        mask_dilate_px=args.mask_dilate,
        neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride,
        subvideo_length=args.subvideo_length,
        resize_ratio=args.resize_ratio,
        device=args.device,
        use_fp16=not args.no_fp16,
        use_sam_local=not args.use_huggingface_sam,
        use_sam3_text=not args.no_sam3_text,  # SAM3 text mode is default
        use_sam3_multigpu=args.multigpu,
        vision_device=args.vision_device,
        language_device=args.language_device,
        output_crf=args.crf,
        output_preset=args.preset,
        cleanup_temp=not args.keep_temp
    )


def print_result(result: PipelineResult, quiet: bool = False):
    """Print pipeline result summary."""
    if quiet:
        if result.success:
            print(result.output_path)
        return

    print("\n" + "=" * 60)
    if result.success:
        print("WATERMARK REMOVAL COMPLETE")
    else:
        print("WATERMARK REMOVAL FAILED")
    print("=" * 60)

    print(f"\nInput:  {result.input_path}")
    print(f"Output: {result.output_path}")

    print(f"\nStatistics:")
    print(f"  Total frames:           {result.total_frames}")
    print(f"  Frames with watermark:  {result.frames_with_watermark}")
    print(f"  Average mask coverage:  {result.average_mask_coverage_pct:.1f}%")
    print(f"  Processing time:        {result.processing_time_seconds:.1f}s")

    print(f"\nAudio validation:")
    status = "OK" if result.audio_synced else "DESYNC"
    print(f"  Status:      {status}")
    print(f"  Drift:       {result.duration_diff_ms:.2f}ms")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.error_message:
        print(f"\nError: {result.error_message}")

    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        setup_logging(args.verbose)

    # Handle preview mode
    if args.preview_frame is not None:
        success = preview_single_frame(args)
        sys.exit(0 if success else 1)

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"

    # Build config and run
    config = build_config(args)
    job = StandaloneWatermarkJob(config)

    if not args.quiet:
        print(f"Removing watermarks from: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Device: {config.device}, FP16: {config.use_fp16}")
        if config.use_sam3_text:
            if config.use_sam3_multigpu:
                print(f"Mode: SAM3 text prompts (MULTI-GPU: vision={config.vision_device}, language={config.language_device})")
            else:
                print(f"Mode: SAM3 text prompts (no YOLO needed)")
        else:
            print(f"Mode: YOLO + SAM2")
        print()

    result = job.run(
        input_video=input_path,
        output_video=output_path,
        temp_dir=args.temp_dir
    )

    # Print result
    print_result(result, args.quiet)

    # Save result metadata
    if result.success and args.save_masks:
        result_json = output_path.parent / f"{output_path.stem}_result.json"
        import json
        with open(result_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        if not args.quiet:
            print(f"Result metadata saved to: {result_json}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
