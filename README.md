# Sora Watermark Remover

Remove animated watermarks from Sora-generated videos using AI-powered detection and inpainting.

**Pipeline**: SAM3 (text-prompted segmentation) → ProPainter (temporal-consistent inpainting)

## Features

- **Text-prompted detection**: Uses SAM3's grounded segmentation to find text/watermarks without manual annotation
- **Temporal consistency**: ProPainter ensures smooth inpainting across video frames
- **Multi-GPU support**: Split SAM3 across GPUs for faster processing on multi-GPU systems
- **Audio preservation**: Original audio is preserved with zero re-encoding
- **Full resolution output**: Processes at reduced resolution for memory efficiency, upscales back to original

## Requirements

- Python 3.10+ (3.12 recommended for SAM3)
- NVIDIA GPU with 8GB+ VRAM (24GB recommended for HD video)
- CUDA 11.8+
- FFmpeg (for video I/O)

### Install FFmpeg

```bash
# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install Gyan.FFmpeg
# Or download from: https://ffmpeg.org/download.html
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sora-watermark-remover.git
cd sora-watermark-remover
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install PyTorch (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install SAM3

```bash
pip install git+https://github.com/anthropics/sam3.git
```

### 5. Install other dependencies

```bash
pip install -r requirements.txt
```

### 6. Clone ProPainter

```bash
mkdir -p vendor
git clone https://github.com/sczhou/ProPainter.git vendor/ProPainter
```

### 7. Download models

Create a `models/` directory and download:

- **SAM3 checkpoint** (`sam3.pt`): ~2.5GB
- **ProPainter weights**: Auto-downloaded on first run to `vendor/ProPainter/weights/`

## Usage

### Basic usage

```bash
python remove_text.py --input video.mp4 --output clean.mp4
```

### Multi-GPU mode (recommended for systems with 2+ GPUs)

```bash
python remove_text.py --input video.mp4 --output clean.mp4 --multigpu
```

### Custom detection prompts

```bash
python remove_text.py --input video.mp4 --output clean.mp4 \
    --target-text "SORA,text,watermark,@username"
```

### Memory-efficient mode (for long/HD videos)

```bash
python remove_text.py --input long_video.mp4 --output clean.mp4 \
    --subvideo-length 40 --resize-ratio 0.5
```

### Preview single frame (for testing detection)

```bash
python remove_text.py --input video.mp4 --preview-frame 100
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | Required | Input video file |
| `--output`, `-o` | `{input}_clean.mp4` | Output video file |
| `--target-text` | `word,text,watermark,letters` | Detection prompts (comma-separated) |
| `--conf-threshold` | `0.15` | Detection confidence threshold |
| `--mask-dilate` | `3` | Pixels to expand mask |
| `--multigpu` | `False` | Split SAM3 across multiple GPUs |
| `--vision-device` | `cuda:0` | GPU for vision backbone |
| `--language-device` | `cuda:1` | GPU for language model |
| `--subvideo-length` | `40` | Frames per chunk (reduce for OOM) |
| `--resize-ratio` | `0.5` | Scale factor for ProPainter (0.5 = half resolution) |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--no-fp16` | `False` | Disable half precision |
| `--crf` | `18` | Output video quality (0-51, lower=better) |
| `--verbose`, `-v` | `False` | Enable debug logging |
| `--keep-temp` | `False` | Keep temporary files |

## Architecture

```
Input Video
    ↓
FFmpeg: Extract Frames (PNG)
    ↓
SAM3: Text-prompted segmentation
    │  Prompts: "word", "text", "watermark", "letters"
    │  Outputs: Binary masks per frame
    ↓
ProPainter: Temporal-consistent video inpainting
    │  Uses optical flow for temporal coherence
    │  Processes in chunks for memory efficiency
    ↓
FFmpeg: Upscale + Stitch with original audio
    ↓
Output Video (watermark removed)
```

## Project Structure

```
sora-watermark-remover/
├── remove_text.py          # CLI entry point
├── src/
│   ├── core/
│   │   ├── sam3_segmenter.py      # SAM3 text-prompted segmentation
│   │   ├── sam3_multigpu.py       # Multi-GPU SAM3 implementation
│   │   ├── propainter.py          # ProPainter video inpainting
│   │   ├── audio_preservation.py  # FFmpeg audio handling
│   │   ├── mask_utils.py          # Mask processing utilities
│   │   └── yolo_detector.py       # Optional YOLO-World detector
│   └── pipeline/
│       └── standalone.py          # Main pipeline orchestration
├── models/                 # Model weights (not in git)
│   └── sam3.pt
├── vendor/                 # External repos (not in git)
│   └── ProPainter/
├── requirements.txt
└── README.md
```

## Performance

Tested on RTX 3090 (24GB) + RTX 3070 (8GB):

| Video | Resolution | Duration | Processing Time |
|-------|------------|----------|-----------------|
| sora1.mp4 | 704x1280 | 10s | ~7 minutes |

- SAM3 detection: ~0.6s per frame
- ProPainter inpainting: ~3 minutes for 291 frames at 0.5x scale

## Memory Requirements

| Resolution | Single GPU | Multi-GPU |
|------------|------------|-----------|
| 720p | 8GB+ | 6GB + 4GB |
| 1080p | 16GB+ | 12GB + 6GB |
| 4K | 24GB+ | Use resize_ratio=0.25 |

## Troubleshooting

### CUDA Out of Memory

- Reduce `--subvideo-length` (try 20 or 30)
- Reduce `--resize-ratio` (try 0.25)
- Use `--multigpu` to split model across GPUs

### No watermarks detected

- Lower `--conf-threshold` (try 0.1)
- Add more prompts to `--target-text`
- Use `--preview-frame` to debug detection

### Blurry output

- Increase `--resize-ratio` (try 0.75 or 1.0)
- Increase `--neighbor-length` for more temporal context

## License

This project uses several components with different licenses:

- **SAM3**: Check Anthropic's license
- **ProPainter**: Non-commercial use (see [ProPainter license](https://github.com/sczhou/ProPainter/blob/main/LICENSE))
- **This codebase**: MIT License

For commercial use, please review the licenses of underlying models.

## Acknowledgments

- [SAM3](https://github.com/anthropics/sam3) - Text-prompted segmentation
- [ProPainter](https://github.com/sczhou/ProPainter) - Video inpainting
- [FFmpeg](https://ffmpeg.org/) - Video processing
