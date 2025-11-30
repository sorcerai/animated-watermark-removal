#!/bin/bash
# Setup script for Animated Watermark Remover - Standalone Pipeline
# Run this on your Windows server (WSL/Git Bash) or adjust for PowerShell

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================="
echo "Animated Watermark Remover Setup"
echo "==================================="

# 1. Create vendor directory
echo ""
echo "[1/4] Setting up vendor directory..."
mkdir -p vendor

# 2. Clone ProPainter
echo ""
echo "[2/4] Cloning ProPainter..."
if [ ! -d "vendor/ProPainter" ]; then
    git clone https://github.com/sczhou/ProPainter.git vendor/ProPainter
    echo "ProPainter cloned successfully"
else
    echo "ProPainter already exists, skipping clone"
fi

# 3. Download ProPainter weights
echo ""
echo "[3/4] Downloading ProPainter weights..."
mkdir -p vendor/ProPainter/weights

PROPAINTER_WEIGHTS=(
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth"
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth"
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"
)

for url in "${PROPAINTER_WEIGHTS[@]}"; do
    filename=$(basename "$url")
    if [ ! -f "vendor/ProPainter/weights/$filename" ]; then
        echo "Downloading $filename..."
        curl -L "$url" -o "vendor/ProPainter/weights/$filename"
    else
        echo "$filename already exists, skipping"
    fi
done

# 4. Download YOLO-World model (if not present)
echo ""
echo "[4/4] Checking YOLO-World model..."
if [ ! -f "models/yolo-world.pt" ] && [ ! -f "models/yolov8l-worldv2.pt" ]; then
    echo "YOLO-World model not found."
    echo "You can download it with: yolo download yolov8l-worldv2"
    echo "Or use ultralytics to auto-download on first run"
    echo ""
    echo "Alternatively, any YOLO-World variant works:"
    echo "  - yolov8s-worldv2.pt (small, faster)"
    echo "  - yolov8m-worldv2.pt (medium)"
    echo "  - yolov8l-worldv2.pt (large, recommended)"
    echo "  - yolov8x-worldv2.pt (extra large, best accuracy)"
else
    echo "YOLO-World model found"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Models available:"
ls -lh models/
echo ""
echo "To test the pipeline:"
echo "  python remove_text.py --input your_video.mp4 --output clean_video.mp4"
echo ""
echo "To preview a single frame first:"
echo "  python remove_text.py --input your_video.mp4 --preview-frame 50"
echo ""
