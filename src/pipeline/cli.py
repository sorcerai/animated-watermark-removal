"""Command-line interface for the animated watermark removal pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from comfyui import ComfyUIClient
from .job import PipelineConfig, SoraWatermarkJob


def _parse_point(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None

    try:
        x_str, y_str = value.split(",")
        return int(x_str.strip()), int(y_str.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "SAM point must be provided as 'x,y' (e.g. 960,540)"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the animated watermark removal pipeline end-to-end.",
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video clip (MP4, 1080p, <= 15s).",
    )

    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path("workflows/sora-removal-production.json"),
        help="Path to the tuned ComfyUI workflow JSON.",
    )

    parser.add_argument(
        "--host",
        default="http://localhost:8188",
        help="ComfyUI host endpoint (default: http://localhost:8188).",
    )

    parser.add_argument(
        "--sam-point",
        type=_parse_point,
        default=None,
        help="Override SAM2 prompt coordinates as 'x,y'.",
    )

    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("working"),
        help="Root directory for temporary job artifacts.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Destination directory for processed videos.",
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store JSON job logs.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum allowed clip duration in seconds (default: 15).",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve the working directory instead of cleaning up.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the source video with the processed output.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak copy when overwriting the input video.",
    )

    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=33.0,
        help="Maximum allowed audio sync delta in milliseconds.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        workflow_path=args.workflow,
        working_root=args.working_dir,
        output_root=args.output_dir,
        logs_root=args.logs_dir,
        keep_temp=args.keep_temp,
        overwrite=args.overwrite,
        backup_original=not args.no_backup,
        max_duration_seconds=args.max_duration,
        audio_tolerance_ms=args.tolerance_ms,
    )

    comfy_client = ComfyUIClient(host=args.host)

    job = SoraWatermarkJob(
        input_video=args.input_video,
        config=config,
        comfy_client=comfy_client,
        sam_point=args.sam_point,
    )

    try:
        summary = job.run()
    except Exception as exc:  # pragma: no cover - CLI guard path
        parser.error(str(exc))
        return 1

    print("\nPipeline completed successfully ✅")
    print(f"Output video: {summary['output_video']}")
    print(f"Audio sync delta: {summary['audio_validation']['duration_diff_ms']} ms")
    print(f"Frames rendered: {summary['frames_saved']}")
    print(f"Log file: {summary['log_file']}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
