#!/usr/bin/env python3
"""
ProPainter Video Inpainting Wrapper

Integrates sczhou/ProPainter for temporal-consistent video inpainting.
Handles long videos via chunked processing with overlapping padding.

Models required (auto-downloaded if not present):
- ProPainter.pth (main inpainting model)
- recurrent_flow_completion.pth (flow completion network)
- raft-things.pth (optical flow model)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProPainterInpainter:
    """
    ProPainter wrapper for temporal-consistent video inpainting.

    Supports two modes:
    1. Subprocess mode: Calls ProPainter's inference script (recommended)
    2. Direct mode: Imports ProPainter modules directly (requires vendor setup)

    The subprocess mode is more reliable as it handles all internal dependencies.
    """

    # ProPainter default parameters
    DEFAULT_NEIGHBOR_LENGTH = 10
    DEFAULT_REF_STRIDE = 10
    DEFAULT_SUBVIDEO_LENGTH = 40  # Reduced from 80 to prevent OOM on HD video

    # Model URLs from ProPainter releases
    MODEL_URLS = {
        "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    }

    def __init__(
        self,
        propainter_dir: Union[str, Path],
        weights_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        use_fp16: bool = True,
        use_subprocess: bool = True
    ):
        """
        Initialize ProPainter inpainter.

        Args:
            propainter_dir: Path to cloned ProPainter repository
            weights_dir: Directory containing model weights (defaults to propainter_dir/weights)
            device: Device to run on ("cuda" or "cpu")
            use_fp16: Use half precision for memory efficiency
            use_subprocess: Use subprocess mode (recommended) vs direct import
        """
        self.propainter_dir = Path(propainter_dir).resolve()
        self.weights_dir = Path(weights_dir) if weights_dir else self.propainter_dir / "weights"
        self.device = device
        self.use_fp16 = use_fp16
        self.use_subprocess = use_subprocess

        # Validate setup
        self._validate_setup()

        # Models (loaded lazily if using direct mode)
        self._raft = None
        self._flow_complete = None
        self._inpaint_model = None

    def _validate_setup(self):
        """Validate ProPainter installation and weights."""
        # Check ProPainter directory
        inference_script = self.propainter_dir / "inference_propainter.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"ProPainter inference script not found at {inference_script}. "
                f"Clone the repo: git clone https://github.com/sczhou/ProPainter.git"
            )

        # Check/create weights directory
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Check for required model files
        missing_models = []
        for model_name in self.MODEL_URLS.keys():
            model_path = self.weights_dir / model_name
            if not model_path.exists():
                missing_models.append(model_name)

        if missing_models:
            logger.warning(
                f"Missing model weights: {missing_models}. "
                f"They will be auto-downloaded on first run, or download manually from: "
                f"https://github.com/sczhou/ProPainter/releases/tag/v0.1.0"
            )

    def download_weights(self):
        """Download missing model weights."""
        try:
            import torch
            from torch.hub import download_url_to_file
        except ImportError:
            raise ImportError("PyTorch required for downloading weights")

        for model_name, url in self.MODEL_URLS.items():
            model_path = self.weights_dir / model_name
            if not model_path.exists():
                logger.info(f"Downloading {model_name}...")
                download_url_to_file(url, str(model_path))
                logger.info(f"Downloaded {model_name} to {model_path}")

    def inpaint_video(
        self,
        frames_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        output_dir: Union[str, Path],
        height: Optional[int] = None,
        width: Optional[int] = None,
        neighbor_length: int = DEFAULT_NEIGHBOR_LENGTH,
        ref_stride: int = DEFAULT_REF_STRIDE,
        subvideo_length: int = DEFAULT_SUBVIDEO_LENGTH,
        resize_ratio: float = 1.0
    ) -> Path:
        """
        Inpaint video frames with temporal consistency.

        Args:
            frames_dir: Directory containing input frames (PNG/JPG)
            masks_dir: Directory containing mask frames (white = inpaint region)
            output_dir: Directory to save inpainted frames
            height: Output height (None = auto from input)
            width: Output width (None = auto from input)
            neighbor_length: Local neighbor frames for propagation
            ref_stride: Global reference frame stride
            subvideo_length: Frames per sub-video chunk (for memory)
            resize_ratio: Resize factor for processing

        Returns:
            Path to output frames directory
        """
        frames_dir = Path(frames_dir)
        masks_dir = Path(masks_dir)
        output_dir = Path(output_dir)

        # Validate inputs
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_subprocess:
            return self._inpaint_subprocess(
                frames_dir, masks_dir, output_dir,
                height, width, neighbor_length, ref_stride,
                subvideo_length, resize_ratio
            )
        else:
            return self._inpaint_direct(
                frames_dir, masks_dir, output_dir,
                height, width, neighbor_length, ref_stride,
                subvideo_length, resize_ratio
            )

    def _inpaint_subprocess(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        height: Optional[int],
        width: Optional[int],
        neighbor_length: int,
        ref_stride: int,
        subvideo_length: int,
        resize_ratio: float
    ) -> Path:
        """Run ProPainter via subprocess (recommended mode)."""
        # Convert to absolute paths (ProPainter runs with different cwd)
        frames_dir = frames_dir.resolve()
        masks_dir = masks_dir.resolve()
        output_dir = output_dir.resolve()

        # Build command
        cmd = [
            sys.executable,
            str(self.propainter_dir / "inference_propainter.py"),
            "--video", str(frames_dir),
            "--mask", str(masks_dir),
            "--output", str(output_dir),
            "--neighbor_length", str(neighbor_length),
            "--ref_stride", str(ref_stride),
            "--subvideo_length", str(subvideo_length),
            "--resize_ratio", str(resize_ratio),
            "--save_frames",  # Save as frames, not video
        ]

        # Optional arguments
        if height is not None:
            cmd.extend(["--height", str(height)])
        if width is not None:
            cmd.extend(["--width", str(width)])
        if self.use_fp16:
            cmd.append("--fp16")

        # Set environment for weights directory
        env = os.environ.copy()
        env["PROPAINTER_WEIGHTS"] = str(self.weights_dir)

        logger.info(f"Running ProPainter: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.propainter_dir),
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            logger.debug(f"ProPainter stdout: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"ProPainter failed: {e.stderr}")
            raise RuntimeError(f"ProPainter inference failed: {e.stderr}")

        # Find output frames (ProPainter creates subfolder with results)
        result_dir = self._find_result_frames(output_dir)
        return result_dir

    def _find_result_frames(self, output_dir: Path) -> Path:
        """Find the directory containing result frames."""
        # ProPainter creates nested structure: output/frames/frames/*.png
        # Search recursively for the deepest directory containing PNG/JPG files

        def find_frames_dir(search_dir: Path, depth: int = 0) -> Optional[Path]:
            if depth > 3:  # Limit search depth
                return None

            # Check this directory for frames
            frames = list(search_dir.glob("*.png")) + list(search_dir.glob("*.jpg"))
            if frames:
                return search_dir

            # Search subdirectories
            try:
                for subdir in search_dir.iterdir():
                    if subdir.is_dir():
                        result = find_frames_dir(subdir, depth + 1)
                        if result:
                            return result
            except PermissionError:
                pass

            return None

        result = find_frames_dir(output_dir)
        if result:
            return result

        raise FileNotFoundError(f"No result frames found in {output_dir}")

    def _inpaint_direct(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_dir: Path,
        height: Optional[int],
        width: Optional[int],
        neighbor_length: int,
        ref_stride: int,
        subvideo_length: int,
        resize_ratio: float
    ) -> Path:
        """
        Run ProPainter via direct Python import.

        This mode imports ProPainter modules directly, which requires
        the ProPainter repo to be properly set up in the Python path.
        """
        # Add ProPainter to path
        if str(self.propainter_dir) not in sys.path:
            sys.path.insert(0, str(self.propainter_dir))

        try:
            import torch
            from core.utils import to_tensors
            from model.modules.flow_comp_raft import RAFT_bi
            from model.recurrent_flow_completion import RecurrentFlowCompleteNet
            from model.propainter import InpaintGenerator
            from utils.download_util import load_file_from_url
        except ImportError as e:
            raise ImportError(
                f"Failed to import ProPainter modules: {e}. "
                f"Ensure ProPainter is properly installed with all dependencies."
            )

        # Load models if not loaded
        if self._inpaint_model is None:
            self._load_models_direct()

        # Read frames
        frames = self._read_frames(frames_dir)
        masks = self._read_masks(masks_dir, len(frames))

        if height is None or width is None:
            h, w = frames[0].shape[:2]
            height = height or h
            width = width or w

        # Ensure dimensions divisible by 8
        height = height // 8 * 8
        width = width // 8 * 8

        # Process video
        result_frames = self._process_video_direct(
            frames, masks, height, width,
            neighbor_length, ref_stride, subvideo_length
        )

        # Save output frames
        for i, frame in enumerate(result_frames):
            output_path = output_dir / f"frame_{i:05d}.png"
            cv2.imwrite(str(output_path), frame)

        return output_dir

    def _load_models_direct(self):
        """Load ProPainter models directly."""
        import torch

        device = torch.device(self.device)

        # Ensure weights exist
        self.download_weights()

        # Load RAFT
        raft_path = self.weights_dir / "raft-things.pth"
        sys.path.insert(0, str(self.propainter_dir))
        from model.modules.flow_comp_raft import RAFT_bi
        self._raft = RAFT_bi(str(raft_path), device)

        # Load flow completion network
        flow_path = self.weights_dir / "recurrent_flow_completion.pth"
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        self._flow_complete = RecurrentFlowCompleteNet(str(flow_path))
        self._flow_complete.to(device)
        self._flow_complete.eval()

        # Load inpainting model
        model_path = self.weights_dir / "ProPainter.pth"
        from model.propainter import InpaintGenerator
        self._inpaint_model = InpaintGenerator(model_path=str(model_path))
        self._inpaint_model.to(device)
        self._inpaint_model.eval()

        if self.use_fp16 and self.device == "cuda":
            self._flow_complete = self._flow_complete.half()
            self._inpaint_model = self._inpaint_model.half()

        logger.info("ProPainter models loaded successfully")

    def _read_frames(self, frames_dir: Path) -> List[np.ndarray]:
        """Read frames from directory."""
        frame_files = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        if not frame_files:
            raise FileNotFoundError(f"No frames found in {frames_dir}")

        frames = []
        for f in frame_files:
            frame = cv2.imread(str(f))
            if frame is not None:
                frames.append(frame)

        logger.info(f"Loaded {len(frames)} frames from {frames_dir}")
        return frames

    def _read_masks(self, masks_dir: Path, num_frames: int) -> List[np.ndarray]:
        """Read masks from directory."""
        mask_files = sorted(
            list(masks_dir.glob("*.png")) +
            list(masks_dir.glob("*.jpg")) +
            list(masks_dir.glob("*.jpeg"))
        )

        if not mask_files:
            raise FileNotFoundError(f"No masks found in {masks_dir}")

        masks = []
        for f in mask_files:
            mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Ensure binary (white = inpaint region)
                mask = np.where(mask > 127, 255, 0).astype(np.uint8)
                masks.append(mask)

        # If single mask provided, repeat for all frames
        if len(masks) == 1 and num_frames > 1:
            masks = masks * num_frames
            logger.info(f"Replicated single mask for {num_frames} frames")

        # If fewer masks than frames, repeat last mask
        while len(masks) < num_frames:
            masks.append(masks[-1])

        logger.info(f"Loaded {len(masks)} masks from {masks_dir}")
        return masks[:num_frames]

    def _process_video_direct(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        height: int,
        width: int,
        neighbor_length: int,
        ref_stride: int,
        subvideo_length: int
    ) -> List[np.ndarray]:
        """
        Process video through ProPainter pipeline.

        This is a simplified implementation. For production use,
        the subprocess mode is recommended.
        """
        import torch

        device = torch.device(self.device)

        # Resize frames and masks
        resized_frames = [cv2.resize(f, (width, height)) for f in frames]
        resized_masks = [cv2.resize(m, (width, height)) for m in masks]

        # Convert to tensors
        frames_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in resized_frames
        ]).to(device)

        masks_tensor = torch.stack([
            torch.from_numpy(m).float() / 255.0
            for m in resized_masks
        ]).unsqueeze(1).to(device)

        if self.use_fp16 and self.device == "cuda":
            frames_tensor = frames_tensor.half()
            masks_tensor = masks_tensor.half()

        # Process in chunks for memory efficiency
        num_frames = len(frames)
        result_frames = []

        for start_idx in range(0, num_frames, subvideo_length):
            end_idx = min(start_idx + subvideo_length, num_frames)

            # Add padding for temporal context
            pad_start = max(0, start_idx - neighbor_length)
            pad_end = min(num_frames, end_idx + neighbor_length)

            chunk_frames = frames_tensor[pad_start:pad_end]
            chunk_masks = masks_tensor[pad_start:pad_end]

            with torch.no_grad():
                # Simplified flow - full pipeline would use RAFT + flow completion
                # For now, use direct inpainting
                inpainted = self._inpaint_model(
                    chunk_frames.unsqueeze(0),
                    chunk_masks.unsqueeze(0)
                )

            # Extract non-padded results
            offset = start_idx - pad_start
            for i in range(end_idx - start_idx):
                frame_tensor = inpainted[0, offset + i]
                frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                result_frames.append(frame_np)

        return result_frames

    def inpaint_single_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Inpaint a single frame (for testing/preview).

        For video, use inpaint_video() for temporal consistency.

        Args:
            frame: BGR image as numpy array
            mask: Binary mask (white = inpaint region)

        Returns:
            Inpainted frame
        """
        # Create temp directories
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frames_dir = tmpdir / "frames"
            masks_dir = tmpdir / "masks"
            output_dir = tmpdir / "output"

            frames_dir.mkdir()
            masks_dir.mkdir()

            # Save frame and mask
            cv2.imwrite(str(frames_dir / "frame_00000.png"), frame)
            cv2.imwrite(str(masks_dir / "mask_00000.png"), mask)

            # Run inpainting
            result_dir = self.inpaint_video(
                frames_dir, masks_dir, output_dir,
                neighbor_length=1, ref_stride=1, subvideo_length=1
            )

            # Read result
            result_files = list(result_dir.glob("*.png")) + list(result_dir.glob("*.jpg"))
            if result_files:
                return cv2.imread(str(result_files[0]))

        raise RuntimeError("Single frame inpainting failed")

    def __repr__(self):
        return (
            f"ProPainterInpainter("
            f"dir={self.propainter_dir.name}, "
            f"device={self.device}, "
            f"fp16={self.use_fp16}, "
            f"mode={'subprocess' if self.use_subprocess else 'direct'})"
        )


class ProPainterInpainterLite:
    """
    Lightweight ProPainter wrapper that uses LaMa for single-frame fallback.

    When ProPainter is not available or for single-frame scenarios,
    falls back to LaMa (Large Mask Inpainting) which is faster but
    lacks temporal consistency.
    """

    def __init__(
        self,
        lama_model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda"
    ):
        """
        Initialize with optional LaMa fallback.

        Args:
            lama_model_path: Path to LaMa model (optional)
            device: Device to run on
        """
        self.lama_model_path = Path(lama_model_path) if lama_model_path else None
        self.device = device
        self._model = None

    def inpaint_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        dilate_kernel: int = 5
    ) -> np.ndarray:
        """
        Inpaint single frame using OpenCV's inpainting (basic fallback).

        Args:
            frame: BGR image as numpy array
            mask: Binary mask (white = inpaint region)
            dilate_kernel: Dilate mask before inpainting

        Returns:
            Inpainted frame
        """
        # Dilate mask slightly for better coverage
        if dilate_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_kernel, dilate_kernel)
            )
            mask = cv2.dilate(mask, kernel)

        # Use OpenCV's Navier-Stokes based inpainting
        result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

        return result

    def inpaint_frames(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Inpaint multiple frames independently (no temporal consistency).

        Args:
            frames: List of BGR images
            masks: List of binary masks

        Returns:
            List of inpainted frames
        """
        results = []
        for frame, mask in zip(frames, masks):
            result = self.inpaint_frame(frame, mask)
            results.append(result)
        return results
