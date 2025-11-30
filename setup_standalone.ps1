# Setup script for Animated Watermark Remover - Standalone Pipeline
# Run this on Windows PowerShell

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Animated Watermark Remover Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# 1. Create vendor directory
Write-Host ""
Write-Host "[1/4] Setting up vendor directory..." -ForegroundColor Yellow
if (-not (Test-Path "vendor")) {
    New-Item -ItemType Directory -Path "vendor" | Out-Null
}

# 2. Clone ProPainter
Write-Host ""
Write-Host "[2/4] Cloning ProPainter..." -ForegroundColor Yellow
if (-not (Test-Path "vendor\ProPainter")) {
    git clone https://github.com/sczhou/ProPainter.git vendor\ProPainter
    Write-Host "ProPainter cloned successfully" -ForegroundColor Green
} else {
    Write-Host "ProPainter already exists, skipping clone" -ForegroundColor Gray
}

# 3. Download ProPainter weights
Write-Host ""
Write-Host "[3/4] Downloading ProPainter weights..." -ForegroundColor Yellow
if (-not (Test-Path "vendor\ProPainter\weights")) {
    New-Item -ItemType Directory -Path "vendor\ProPainter\weights" | Out-Null
}

$ProPainterWeights = @(
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
    "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth"
)

foreach ($url in $ProPainterWeights) {
    $filename = Split-Path -Leaf $url
    $filepath = "vendor\ProPainter\weights\$filename"
    if (-not (Test-Path $filepath)) {
        Write-Host "Downloading $filename..."
        Invoke-WebRequest -Uri $url -OutFile $filepath
    } else {
        Write-Host "$filename already exists, skipping" -ForegroundColor Gray
    }
}

# 4. Check YOLO-World model
Write-Host ""
Write-Host "[4/4] Checking YOLO-World model..." -ForegroundColor Yellow
$yoloExists = (Test-Path "models\yolo-world.pt") -or (Test-Path "models\yolov8l-worldv2.pt")
if (-not $yoloExists) {
    Write-Host "YOLO-World model not found." -ForegroundColor Yellow
    Write-Host "You can download it with: yolo download yolov8l-worldv2" -ForegroundColor Gray
    Write-Host "Or use ultralytics to auto-download on first run" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Alternatively, any YOLO-World variant works:" -ForegroundColor Gray
    Write-Host "  - yolov8s-worldv2.pt (small, faster)" -ForegroundColor Gray
    Write-Host "  - yolov8m-worldv2.pt (medium)" -ForegroundColor Gray
    Write-Host "  - yolov8l-worldv2.pt (large, recommended)" -ForegroundColor Gray
    Write-Host "  - yolov8x-worldv2.pt (extra large, best accuracy)" -ForegroundColor Gray
} else {
    Write-Host "YOLO-World model found" -ForegroundColor Green
}

# 5. Install Python dependencies
Write-Host ""
Write-Host "[5/5] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install SAM2 from GitHub
Write-Host "Installing SAM2..."
pip install git+https://github.com/facebookresearch/sam2.git

Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "Models available:" -ForegroundColor Cyan
Get-ChildItem -Path "models" | Format-Table Name, Length
Write-Host ""
Write-Host "To test the pipeline:" -ForegroundColor Cyan
Write-Host "  python remove_text.py --input your_video.mp4 --output clean_video.mp4"
Write-Host ""
Write-Host "To preview a single frame first:" -ForegroundColor Cyan
Write-Host "  python remove_text.py --input your_video.mp4 --preview-frame 50"
Write-Host ""
