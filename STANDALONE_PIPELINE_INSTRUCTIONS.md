# Sora Watermark Remover - Standalone Pipeline Instructions

## Overview

This is a standalone Python pipeline for removing animated Sora watermarks from videos. It replaces the previous ComfyUI workflow approach with a direct Python implementation.

**Pipeline Flow:**
```
Input Video → Extract Frames → YOLO-World (detect bbox) → SAM2 (segment mask) → ProPainter (inpaint) → Stitch + Audio → Output
```

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| YOLO-World | `src/core/yolo_detector.py` | Open-vocabulary detection of "SORA", "@username", watermark text |
| SAM2 | `src/core/sam3_segmenter.py` | Precise segmentation of text + glow region from bbox prompt |
| ProPainter | `src/core/propainter.py` | Temporal-consistent video inpainting |
| Mask Utils | `src/core/mask_utils.py` | Bbox expansion, mask dilation, merging |
| Audio | `src/core/audio_preservation.py` | FFmpeg wrapper for frame extraction and audio sync |
| Pipeline | `src/pipeline/standalone.py` | Orchestrates all components end-to-end |
| CLI | `remove_text.py` | Command-line interface |

### Key Design Decisions

1. **No hardcoded masks** - Automatic detection per frame
2. **Bbox expansion** - Captures glow/animation effects around text
3. **Mask dilation** - Ensures complete watermark coverage
4. **Temporal consistency** - ProPainter uses optical flow for smooth inpainting
5. **Audio preservation** - Original audio track maintained with sync validation

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- FFmpeg installed and in PATH
- Git

### Step 1: Run Setup Script

**Windows (PowerShell):**
```powershell
cd F:\sora-watermark-remover
.\setup_standalone.ps1
```

**Linux/WSL (Bash):**
```bash
cd /path/to/sora-watermark-remover
chmod +x setup_standalone.sh
./setup_standalone.sh
```

The setup script will:
1. Clone ProPainter repository to `vendor/ProPainter`
2. Download ProPainter weights (~1.5GB total)
3. Install Python dependencies
4. Install SAM2 from GitHub

### Step 2: Verify Models

After setup, verify these models exist:

```
models/
├── sam2_hiera_large.pt      # SAM2 segmentation (already present)
└── yolov8l-worldv2.pt       # YOLO-World detection (auto-downloads on first run)

vendor/ProPainter/weights/
├── ProPainter.pth           # Main inpainting model
├── recurrent_flow_completion.pth  # Flow completion
└── raft-things.pth          # Optical flow (RAFT)
```

### Step 3: Manual Dependency Install (if setup script fails)

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy Pillow

# SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# ProPainter dependencies
cd vendor/ProPainter
pip install -r requirements.txt
cd ../..
```

---

## Usage

### Basic Usage

```bash
python remove_text.py --input sora_video.mp4 --output clean_video.mp4
```

### Preview Single Frame (Recommended First Step)

Test detection and segmentation on a single frame before processing entire video:

```bash
python remove_text.py --input sora_video.mp4 --preview-frame 50
```

This creates a preview folder with:
- `frame_000050_original.png` - Original frame
- `frame_000050_mask.png` - Detected watermark mask
- `frame_000050_result.png` - Inpainted result
- `frame_000050_comparison.png` - Side-by-side comparison

### Full Command Reference

```bash
python remove_text.py \
    --input video.mp4 \
    --output clean.mp4 \
    --yolo-model models/yolov8l-worldv2.pt \
    --sam-model models/sam2_hiera_large.pt \
    --sam-config sam2_hiera_l.yaml \
    --propainter-dir vendor/ProPainter \
    --target-text "SORA,text,watermark,logo,username,brand text" \
    --conf-threshold 0.25 \
    --bbox-expand 10 \
    --mask-dilate 3 \
    --neighbor-length 10 \
    --ref-stride 10 \
    --subvideo-length 80 \
    --device cuda \
    --crf 18 \
    --preset medium
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | (required) | Input video file |
| `--output`, `-o` | `{input}_clean.mp4` | Output video file |
| `--yolo-model` | `models/yolo-world.pt` | YOLO-World weights path |
| `--sam-model` | `models/SAM3.pt` | SAM2 checkpoint path |
| `--sam-config` | `sam2_hiera_l.yaml` | SAM2 config name |
| `--propainter-dir` | `vendor/ProPainter` | ProPainter repo path |
| `--target-text` | `SORA,text,watermark,...` | Detection targets (comma-separated) |
| `--conf-threshold` | `0.25` | Detection confidence threshold |
| `--bbox-expand` | `10` | Pixels to expand bbox for glow capture |
| `--mask-dilate` | `3` | Pixels to dilate mask |
| `--neighbor-length` | `10` | ProPainter local neighbor frames |
| `--ref-stride` | `10` | ProPainter global reference stride |
| `--subvideo-length` | `80` | Frames per chunk (memory management) |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--no-fp16` | `False` | Disable half-precision inference |
| `--crf` | `18` | Output quality (0-51, lower=better) |
| `--preset` | `medium` | Encoding speed preset |
| `--preview-frame` | `None` | Process single frame for testing |
| `--keep-temp` | `False` | Keep temporary files |
| `--verbose`, `-v` | `False` | Enable debug logging |

---

## Troubleshooting

### Issue: YOLO-World not detecting watermark

**Symptoms:** No watermark detected, empty masks

**Solutions:**
1. Lower confidence threshold:
   ```bash
   --conf-threshold 0.15
   ```
2. Add more detection targets:
   ```bash
   --target-text "SORA,sora,text,watermark,logo,username,@,brand,overlay"
   ```
3. Try preview mode to debug:
   ```bash
   --preview-frame 50 --verbose
   ```

### Issue: Incomplete watermark removal (glow/edges visible)

**Symptoms:** Text removed but glow artifacts remain

**Solutions:**
1. Increase bbox expansion:
   ```bash
   --bbox-expand 15
   ```
2. Increase mask dilation:
   ```bash
   --mask-dilate 5
   ```

### Issue: CUDA out of memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce subvideo length:
   ```bash
   --subvideo-length 40
   ```
2. Reduce neighbor length:
   ```bash
   --neighbor-length 5
   ```
3. Disable fp16 (uses more memory but more stable):
   ```bash
   --no-fp16
   ```
4. Use CPU (slow but works):
   ```bash
   --device cpu
   ```

### Issue: ProPainter not found

**Symptoms:** `FileNotFoundError: vendor/ProPainter not found`

**Solution:** Run setup script or manually clone:
```bash
git clone https://github.com/sczhou/ProPainter.git vendor/ProPainter
```

### Issue: SAM2 import error

**Symptoms:** `ModuleNotFoundError: No module named 'sam2'`

**Solution:** Install SAM2 from GitHub:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### Issue: Audio desync

**Symptoms:** Output video audio doesn't match video

**Solutions:**
1. Check FFmpeg is installed: `ffmpeg -version`
2. Verify input video isn't corrupted
3. Check the result metadata for `duration_diff_ms`

---

## Python API Usage

For programmatic use without CLI:

```python
from src.pipeline.standalone import StandaloneWatermarkJob, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    yolo_model_path="models/yolov8l-worldv2.pt",
    sam_model_path="models/sam2_hiera_large.pt",
    sam_model_cfg="sam2_hiera_l.yaml",
    propainter_dir="vendor/ProPainter",
    detection_classes=["SORA", "text", "watermark", "logo", "username"],
    detection_conf_threshold=0.25,
    bbox_expand_px=10,
    mask_dilate_px=3,
    device="cuda",
    use_fp16=True
)

# Create job and run
job = StandaloneWatermarkJob(config)
result = job.run(
    input_video="input.mp4",
    output_video="output.mp4"
)

# Check result
if result.success:
    print(f"Success! Output: {result.output_path}")
    print(f"Frames processed: {result.total_frames}")
    print(f"Frames with watermark: {result.frames_with_watermark}")
    print(f"Processing time: {result.processing_time_seconds:.1f}s")
else:
    print(f"Failed: {result.error_message}")
```

### Process Single Frame

```python
import cv2
from src.pipeline.standalone import StandaloneWatermarkJob, PipelineConfig

config = PipelineConfig(device="cuda")
job = StandaloneWatermarkJob(config)

# Load frame
frame = cv2.imread("frame.png")

# Process with mask output
result_frame, mask = job.process_single_frame(frame, return_mask=True)

# Save results
cv2.imwrite("result.png", result_frame)
cv2.imwrite("mask.png", mask)
```

---

## File Structure

```
sora-watermark-remover/
├── remove_text.py                 # CLI entry point
├── requirements.txt               # Python dependencies
├── setup_standalone.sh            # Bash setup script
├── setup_standalone.ps1           # PowerShell setup script
├── STANDALONE_PIPELINE_INSTRUCTIONS.md  # This file
│
├── models/
│   ├── sam2_hiera_large.pt        # SAM2 segmentation model
│   └── yolov8l-worldv2.pt         # YOLO-World detection (auto-downloads)
│
├── vendor/
│   └── ProPainter/                # ProPainter repository (cloned by setup)
│       ├── weights/
│       │   ├── ProPainter.pth
│       │   ├── recurrent_flow_completion.pth
│       │   └── raft-things.pth
│       └── ...
│
└── src/
    ├── core/
    │   ├── __init__.py
    │   ├── audio_preservation.py  # FFmpeg wrapper
    │   ├── mask_utils.py          # Mask manipulation utilities
    │   ├── yolo_detector.py       # YOLO-World detector
    │   ├── sam3_segmenter.py      # SAM2 segmenter
    │   └── propainter.py          # ProPainter inpainter
    │
    └── pipeline/
        ├── __init__.py
        ├── standalone.py          # Main pipeline orchestration
        └── job.py                 # Legacy ComfyUI pipeline (unused)
```

---

## Performance Expectations

| Video Length | Resolution | VRAM | Approx. Time |
|--------------|------------|------|--------------|
| 10 seconds | 1080p | 8GB | 2-5 minutes |
| 30 seconds | 1080p | 8GB | 5-15 minutes |
| 60 seconds | 1080p | 12GB | 15-30 minutes |
| 60 seconds | 4K | 24GB | 30-60 minutes |

**Tips for faster processing:**
- Use `--preset fast` or `--preset veryfast` for encoding
- Reduce `--subvideo-length` for less memory but more chunks
- Use smaller YOLO model: `yolov8s-worldv2.pt` instead of `l` variant

---

## Model Information

### YOLO-World (Detection)
- **Purpose:** Open-vocabulary object detection
- **Detects:** Text classes like "SORA", "watermark", "logo", "@username"
- **Output:** Bounding boxes with confidence scores
- **Variants:** `s` (small/fast), `m` (medium), `l` (large/recommended), `x` (extra large)

### SAM2 (Segmentation)
- **Purpose:** Precise mask segmentation from bbox prompts
- **Input:** Image + bounding box
- **Output:** Binary mask capturing exact watermark region including glow
- **Config:** `sam2_hiera_l.yaml` for large model

### ProPainter (Inpainting)
- **Purpose:** Temporal-consistent video inpainting
- **Method:** RAFT optical flow → flow completion → image propagation → transformer refinement
- **Key params:**
  - `neighbor_length`: Local temporal window (default: 10)
  - `ref_stride`: Global reference sampling (default: 10)
  - `subvideo_length`: Chunk size for memory (default: 80)

---

## Quick Start Checklist

1. [ ] Run `setup_standalone.ps1` (Windows) or `setup_standalone.sh` (Linux)
2. [ ] Verify `vendor/ProPainter/weights/` has 3 `.pth` files
3. [ ] Verify `models/sam2_hiera_large.pt` exists
4. [ ] Test with preview: `python remove_text.py --input test.mp4 --preview-frame 50`
5. [ ] Check preview output images for correct detection
6. [ ] Run full pipeline: `python remove_text.py --input test.mp4 --output clean.mp4`
7. [ ] Verify output video plays correctly with audio

---

## Support

If issues persist after troubleshooting:
1. Run with `--verbose` flag to get detailed logs
2. Check GPU memory with `nvidia-smi`
3. Verify all model files exist and aren't corrupted
4. Test each component individually using Python API
