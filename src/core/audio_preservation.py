#!/usr/bin/env python3
"""
Audio Preservation Pipeline for Video Inpainting

Handles:
1. Frame extraction (preserving original audio separately)
2. Frame processing (your inpainting workflow)
3. Re-encoding with perfect audio sync

Critical for SaaS: Audio must be bit-perfect copy of original
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict
import shutil


class AudioPreservationPipeline:
    """
    Manages video processing while preserving audio perfectly

    Usage:
        pipeline = AudioPreservationPipeline()

        # Extract frames for processing
        frames_dir = pipeline.extract_frames("input.mp4")

        # [Your inpainting happens here on frames_dir]
        # process_frames(frames_dir, output_dir)

        # Stitch back with original audio
        pipeline.stitch_with_audio(
            frames_dir=output_dir,
            original_video="input.mp4",
            output_video="output.mp4"
        )
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

        # Verify ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"],
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg not found. Install: winget install Gyan.FFmpeg"
            )

    def get_video_info(self, video_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        quality: int = 2  # 1-31, lower is higher quality
    ) -> Path:
        """
        Extract video frames at high quality

        Args:
            video_path: Input video file
            output_dir: Where to save frames (default: temp/frames)
            quality: JPEG quality (1=best, 31=worst)

        Returns:
            Path to directory containing frames (named %06d.png)
        """
        video_path = Path(video_path)

        if output_dir is None:
            output_dir = self.temp_dir / f"frames_{video_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video FPS for later stitching
        info = self.get_video_info(str(video_path))
        video_stream = next(
            (s for s in info["streams"] if s["codec_type"] == "video"),
            None
        )

        if video_stream:
            fps_str = video_stream.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den

            # Save FPS metadata for stitching
            (output_dir / "metadata.json").write_text(
                json.dumps({"fps": fps, "original_video": str(video_path)})
            )

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-qscale:v", str(quality),  # High quality frames
            str(output_dir / "%06d.png")  # 000001.png, 000002.png, ...
        ]

        print(f"Extracting frames from {video_path.name}...")
        subprocess.run(cmd, check=True)

        frame_count = len(list(output_dir.glob("*.png")))
        print(f"✓ Extracted {frame_count} frames to {output_dir}")

        return output_dir

    def stitch_with_audio(
        self,
        frames_dir: str,
        original_video: str,
        output_video: str,
        fps: Optional[float] = None,
        crf: int = 18,  # 0-51, lower is higher quality
        preset: str = "medium",  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        target_size: Optional[tuple] = None  # (width, height) to scale output to
    ) -> Path:
        """
        Stitch processed frames back into video with ORIGINAL audio

        Args:
            frames_dir: Directory containing processed frames (%06d.png)
            original_video: Original video (for audio track)
            output_video: Output path
            fps: Frame rate (auto-detected from metadata if None)
            crf: Video quality (18=high, 23=default, 28=low)
            preset: Encoding speed vs size tradeoff
            target_size: Optional (width, height) to scale output to original resolution

        Returns:
            Path to output video
        """
        frames_dir = Path(frames_dir)
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Try to load FPS from metadata
        metadata_path = frames_dir / "metadata.json"
        if fps is None and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            fps = metadata.get("fps", 30.0)
        elif fps is None:
            fps = 30.0  # Default fallback

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-r", str(fps),  # Input frame rate
            "-i", str(frames_dir / "%06d.png"),  # Input frames
            "-i", str(original_video),  # Original video for audio
            "-map", "0:v",  # Video from processed frames
            "-map", "1:a",  # Audio from original video
        ]

        # Add scale filter if target size specified
        if target_size:
            width, height = target_size
            cmd.extend(["-vf", f"scale={width}:{height}:flags=lanczos"])

        cmd.extend([
            "-c:v", "libx264",  # H.264 codec
            "-crf", str(crf),  # Quality
            "-preset", preset,  # Speed
            "-c:a", "copy",  # COPY audio (no re-encoding = perfect preservation)
            "-movflags", "+faststart",  # Web streaming optimization
            str(output_video)
        ])

        print(f"Stitching {len(list(frames_dir.glob('*.png')))} frames with audio...")
        subprocess.run(cmd, check=True)

        print(f"✓ Created {output_video} with preserved audio")

        return output_video

    def validate_audio_sync(
        self,
        original_video: str,
        processed_video: str,
        tolerance_ms: float = 33.0  # 1 frame at 30fps
    ) -> Dict:
        """
        Verify audio is properly synced between original and processed

        Args:
            original_video: Original video path
            processed_video: Processed video path
            tolerance_ms: Maximum allowed duration difference

        Returns:
            Dict with validation results
        """
        orig_info = self.get_video_info(original_video)
        proc_info = self.get_video_info(processed_video)

        # Get durations
        orig_duration = float(orig_info["format"]["duration"])
        proc_duration = float(proc_info["format"]["duration"])

        duration_diff_ms = abs(orig_duration - proc_duration) * 1000

        # Check audio streams exist
        orig_audio = any(
            s["codec_type"] == "audio"
            for s in orig_info["streams"]
        )
        proc_audio = any(
            s["codec_type"] == "audio"
            for s in proc_info["streams"]
        )

        is_synced = duration_diff_ms <= tolerance_ms

        result = {
            "synced": is_synced and orig_audio and proc_audio,
            "duration_diff_ms": round(duration_diff_ms, 2),
            "tolerance_ms": tolerance_ms,
            "original_duration": orig_duration,
            "processed_duration": proc_duration,
            "original_has_audio": orig_audio,
            "processed_has_audio": proc_audio
        }

        if result["synced"]:
            print(f"✓ Audio sync validated ({duration_diff_ms:.2f}ms difference)")
        else:
            print(f"✗ Audio desync detected ({duration_diff_ms:.2f}ms difference)")
            if not proc_audio:
                print("  ⚠ Processed video has no audio track!")

        return result

    def cleanup(self, frames_dir: Optional[str] = None):
        """Remove temporary frames to save disk space"""
        if frames_dir:
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"✓ Cleaned up {frames_dir}")
        else:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"✓ Cleaned up {self.temp_dir}")


def example_usage():
    """Example workflow"""
    pipeline = AudioPreservationPipeline()

    # Step 1: Extract frames
    frames_dir = pipeline.extract_frames("input.mp4")

    # Step 2: [YOUR INPAINTING HAPPENS HERE]
    # For now, we'll just simulate by copying frames
    processed_dir = Path("temp/processed")
    processed_dir.mkdir(exist_ok=True)
    for frame in frames_dir.glob("*.png"):
        shutil.copy(frame, processed_dir / frame.name)

    # Step 3: Stitch back with audio
    output = pipeline.stitch_with_audio(
        frames_dir=processed_dir,
        original_video="input.mp4",
        output_video="output.mp4"
    )

    # Step 4: Validate audio sync
    validation = pipeline.validate_audio_sync("input.mp4", str(output))

    # Step 5: Cleanup temporary files
    if validation["synced"]:
        pipeline.cleanup(frames_dir)
        pipeline.cleanup(processed_dir)

    return validation


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python audio_preservation.py <input_video> <output_video>")
        print("Example: python audio_preservation.py input.mp4 output.mp4")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    pipeline = AudioPreservationPipeline()

    # Extract
    frames_dir = pipeline.extract_frames(input_video)

    print("\n⚠ Now process frames in:", frames_dir)
    print("   Then run with --stitch flag to continue\n")

    # For full automation, you'd call your inpainting here
    # For manual testing, user processes frames then runs again
