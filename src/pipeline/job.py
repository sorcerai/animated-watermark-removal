"""High-level orchestration for the Sora watermark removal pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core import AudioPreservationPipeline
from comfyui import ComfyUIClient


@dataclass
class PipelineConfig:
    """Configuration for running the watermark removal pipeline."""

    workflow_path: Path = Path("workflows/sora-removal-production.json")
    working_root: Path = Path("working")
    output_root: Path = Path("output")
    logs_root: Path = Path("logs")
    keep_temp: bool = False
    overwrite: bool = False
    backup_original: bool = True
    max_duration_seconds: float = 15.0
    expected_resolution: Tuple[int, int] = (1920, 1080)
    audio_tolerance_ms: float = 33.0
    min_frame_count: int = 5

    def ensure_directories(self) -> None:
        """Create required root directories if they do not already exist."""

        for root in (self.working_root, self.output_root, self.logs_root):
            root.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelinePaths:
    """Concrete filesystem locations for a single job run."""

    job_root: Path
    comfy_output: Path
    processed_frames: Path
    output_video: Path
    log_file: Path
    backup_path: Optional[Path] = None


class PipelineValidationError(RuntimeError):
    """Raised when the input video fails pre-flight validation."""


class SoraWatermarkJob:
    """End-to-end orchestration of the Sora watermark removal workflow."""

    def __init__(
        self,
        input_video: Path,
        config: PipelineConfig,
        comfy_client: Optional[ComfyUIClient] = None,
        audio_pipeline: Optional[AudioPreservationPipeline] = None,
        sam_point: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.input_video = Path(input_video).expanduser().resolve()
        self.config = config
        self.sam_point = sam_point
        self.config.ensure_directories()

        self.audio_pipeline = audio_pipeline or AudioPreservationPipeline(
            temp_dir=self.config.working_root / "audio_temp"
        )
        self.comfy_client = comfy_client or ComfyUIClient()

        self._video_metadata: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Execute the watermark removal job.

        Returns
        -------
        Dict
            Summary payload capturing paths, timings, and validation results.
        """

        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")

        if not self.config.workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow not found at {self.config.workflow_path}. "
                "Import the tuned ComfyUI workflow before running."
            )

        self._video_metadata = self._validate_video()
        paths = self._prepare_paths()

        comfy_result = self._invoke_comfy_workflow()
        frame_paths = self._download_processed_frames(paths, comfy_result)

        if len(frame_paths) < self.config.min_frame_count:
            raise RuntimeError(
                "ComfyUI returned fewer frames than expected. "
                f"Got {len(frame_paths)}, expected at least {self.config.min_frame_count}."
            )

        processed_video = self.audio_pipeline.stitch_with_audio(
            frames_dir=str(paths.processed_frames),
            original_video=str(self.input_video),
            output_video=str(paths.output_video),
            fps=self._video_metadata["fps"],
        )

        audio_validation = self.audio_pipeline.validate_audio_sync(
            original_video=str(self.input_video),
            processed_video=str(processed_video),
            tolerance_ms=self.config.audio_tolerance_ms,
        )

        log_payload = self._build_log_payload(
            comfy_result=comfy_result,
            frames=frame_paths,
            audio_validation=audio_validation,
            paths=paths,
        )
        self._write_log(paths.log_file, log_payload)

        if not self.config.keep_temp:
            self._cleanup(paths)

        return log_payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_video(self) -> Dict:
        info = self.audio_pipeline.get_video_info(str(self.input_video))

        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )

        if not video_stream:
            raise PipelineValidationError("No video stream detected in input file")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        expected_width, expected_height = self.config.expected_resolution
        if (width, height) != (expected_width, expected_height):
            raise PipelineValidationError(
                "Input video resolution mismatch. "
                f"Expected {expected_width}x{expected_height}, got {width}x{height}."
            )

        duration = float(info.get("format", {}).get("duration", 0.0))
        if duration > self.config.max_duration_seconds:
            raise PipelineValidationError(
                "Input video exceeds maximum supported duration. "
                f"Max {self.config.max_duration_seconds}s, got {duration:.2f}s."
            )

        fps = self._extract_fps(video_stream)

        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
        }

    def _extract_fps(self, video_stream: Dict) -> float:
        fps_str = (
            video_stream.get("avg_frame_rate")
            or video_stream.get("r_frame_rate")
            or "30/1"
        )

        num, den = fps_str.split("/")
        try:
            return float(num) / float(den)
        except ZeroDivisionError as exc:  # pragma: no cover - defensive guard
            raise PipelineValidationError("Invalid FPS metadata from ffprobe") from exc

    def _prepare_paths(self) -> PipelinePaths:
        job_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        job_root = self.config.working_root / f"{self.input_video.stem}-{job_id}"
        comfy_output = job_root / "comfy_output"
        processed_frames = job_root / "frames_processed"

        job_root.mkdir(parents=True, exist_ok=True)
        comfy_output.mkdir(parents=True, exist_ok=True)
        processed_frames.mkdir(parents=True, exist_ok=True)

        if self.config.overwrite:
            output_video = self.input_video
            backup_path = None
            if self.config.backup_original:
                backup_path = self.input_video.with_suffix(
                    f".bak{self.input_video.suffix}"
                )
                shutil.copy2(self.input_video, backup_path)
        else:
            output_video = self.config.output_root / f"{self.input_video.stem}_clean.mp4"
            backup_path = None

        self.config.logs_root.mkdir(parents=True, exist_ok=True)
        log_file = self.config.logs_root / f"{self.input_video.stem}-{job_id}.json"

        return PipelinePaths(
            job_root=job_root,
            comfy_output=comfy_output,
            processed_frames=processed_frames,
            output_video=output_video,
            log_file=log_file,
            backup_path=backup_path,
        )

    def _invoke_comfy_workflow(self) -> Dict:
        default_point = (
            self.config.expected_resolution[0] // 2,
            self.config.expected_resolution[1] // 2,
        )

        result = self.comfy_client.process_video(
            video_path=str(self.input_video),
            workflow_path=str(self.config.workflow_path),
            sam_point=self.sam_point or default_point,
            callback=lambda progress: print(f"ComfyUI progress: {progress}%"),
        )

        if result.get("status") != "success":
            raise RuntimeError(
                "ComfyUI workflow execution failed. "
                f"Prompt ID: {result.get('prompt_id')}"
            )

        return result

    def _download_processed_frames(
        self, paths: PipelinePaths, comfy_result: Dict
    ) -> List[Path]:
        frames: List[Path] = []

        output_entries = comfy_result.get("output_frames") or []
        if not output_entries:
            raise RuntimeError(
                "ComfyUI response did not include output frames. "
                "Ensure the workflow ends with a SaveImage/SaveVideo node."
            )

        for entry in sorted(output_entries, key=lambda item: item.get("filename", "")):
            filename = Path(entry.get("filename", "frame.png")).name
            subfolder = entry.get("subfolder", "")
            folder_type = entry.get("type", "output")

            data = self.comfy_client.download_output(
                filename=filename,
                subfolder=subfolder,
                folder_type=folder_type,
            )

            target_path = paths.processed_frames / filename
            target_path.write_bytes(data)
            frames.append(target_path)

        # Persist metadata needed for stitching
        metadata = {
            "fps": self._video_metadata["fps"],
            "original_video": str(self.input_video),
            "prompt_id": comfy_result.get("prompt_id"),
            "frame_count": len(frames),
        }
        (paths.processed_frames / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

        return frames

    def _build_log_payload(
        self,
        comfy_result: Dict,
        frames: List[Path],
        audio_validation: Dict,
        paths: PipelinePaths,
    ) -> Dict:
        return {
            "input_video": str(self.input_video),
            "output_video": str(paths.output_video),
            "backup": str(paths.backup_path) if paths.backup_path else None,
            "workflow": str(self.config.workflow_path),
            "sam_point": self.sam_point,
            "video_metadata": self._video_metadata,
            "comfy_prompt_id": comfy_result.get("prompt_id"),
            "comfy_processing_time": comfy_result.get("processing_time"),
            "frames_saved": len(frames),
            "audio_validation": audio_validation,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "log_file": str(paths.log_file),
        }

    def _write_log(self, log_file: Path, payload: Dict) -> None:
        log_file.write_text(json.dumps(payload, indent=2))
        print(f"✓ Wrote job log to {log_file}")

    def _cleanup(self, paths: PipelinePaths) -> None:
        shutil.rmtree(paths.job_root, ignore_errors=True)
        audio_temp = self.config.working_root / "audio_temp"
        shutil.rmtree(audio_temp, ignore_errors=True)
        print(f"✓ Cleaned workspace {paths.job_root}")
