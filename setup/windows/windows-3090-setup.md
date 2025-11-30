# Windows 3090 ComfyUI Setup Guide

## Prerequisites Check

```powershell
# Verify GPU
nvidia-smi

# Should show: RTX 3090, 24GB VRAM, Driver 5xx.xx+
# If not, install latest NVIDIA drivers from:
# https://www.nvidia.com/Download/index.aspx
```

## Step 1: ComfyUI Portable Installation

### Download ComfyUI Portable (Recommended)

```powershell
# Create project directory
New-Item -Path "C:\sora-watermark-saas" -ItemType Directory -Force
cd C:\sora-watermark-saas

# Download ComfyUI Portable for Windows (includes Python, PyTorch with CUDA)
# Go to: https://github.com/comfyanonymous/ComfyUI/releases
# Download: ComfyUI_windows_portable_nvidia_cu121_or_cpu.7z (~4GB)

# Extract to C:\sora-watermark-saas\ComfyUI
# After extraction, your structure should be:
# C:\sora-watermark-saas\ComfyUI\
#   ├── python_embeded/
#   ├── ComfyUI/
#   ├── run_nvidia_gpu.bat
#   └── run_cpu.bat
```

### Test ComfyUI

```powershell
cd C:\sora-watermark-saas\ComfyUI
.\run_nvidia_gpu.bat

# Browser should open to http://127.0.0.1:8188
# You should see ComfyUI interface
# Close it for now (Ctrl+C in terminal)
```

## Step 2: Install Custom Nodes

```powershell
cd C:\sora-watermark-saas\ComfyUI\ComfyUI\custom_nodes

# 1. DiffuEraser (core inpainting)
git clone https://github.com/ShmuelRonen/ComfyUI_DiffuEraser.git
cd ComfyUI_DiffuEraser
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..

# 2. Video Helper Suite
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..

# 3. SAM 2 (segmentation - THE CRITICAL ONE)
git clone https://github.com/kijai/ComfyUI-SAM2.git
cd ComfyUI-SAM2
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..

# 4. KJNodes (utilities)
git clone https://github.com/kijai/ComfyUI-KJNodes.git
cd ComfyUI-KJNodes
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..

# 5. LayerStyle (masking)
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
cd ComfyUI_LayerStyle
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..

# 6. Easy-Use (workflow helpers)
git clone https://github.com/yolain/ComfyUI-Easy-Use.git
cd ComfyUI-Easy-Use
..\..\python_embeded\python.exe -m pip install -r requirements.txt
cd ..
```

## Step 3: Download Models

### Directory Structure
```powershell
# Create model directories
New-Item -Path "C:\sora-watermark-saas\ComfyUI\ComfyUI\models\checkpoints" -ItemType Directory -Force
New-Item -Path "C:\sora-watermark-saas\ComfyUI\ComfyUI\models\sam2" -ItemType Directory -Force
New-Item -Path "C:\sora-watermark-saas\ComfyUI\ComfyUI\models\diffueraser" -ItemType Directory -Force
```

### Model Downloads

**DiffuEraser Checkpoint** (~1.8GB)
```
Download: https://huggingface.co/YihanLi/DiffuEraser/resolve/main/pcm_sd15_smallcfg_2step_converted.safetensors
Place in: C:\sora-watermark-saas\ComfyUI\ComfyUI\models\checkpoints\
```

**Base Model** (~2GB)
```
Download: https://civitai.com/models/4201/realistic-vision-v51
File: realisticVisionV51_v51VAE.safetensors
Place in: C:\sora-watermark-saas\ComfyUI\ComfyUI\models\checkpoints\
```

**SAM 2 Base Model** (~400MB) - Recommended for 24GB VRAM
```
Download: https://huggingface.co/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt
Place in: C:\sora-watermark-saas\ComfyUI\ComfyUI\models\sam2\
```

Alternatively, SAM 2 Large (~800MB) for best quality:
```
Download: https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt
Place in: C:\sora-watermark-saas\ComfyUI\ComfyUI\models\sam2\
```

### Download Script (PowerShell)

```powershell
# Save as download-models.ps1
$models = @(
    @{
        Name = "DiffuEraser"
        URL = "https://huggingface.co/YihanLi/DiffuEraser/resolve/main/pcm_sd15_smallcfg_2step_converted.safetensors"
        Path = "C:\sora-watermark-saas\ComfyUI\ComfyUI\models\checkpoints\pcm_sd15_smallcfg_2step_converted.safetensors"
    },
    @{
        Name = "SAM2-Base"
        URL = "https://huggingface.co/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt"
        Path = "C:\sora-watermark-saas\ComfyUI\ComfyUI\models\sam2\sam2_hiera_base_plus.pt"
    }
)

foreach ($model in $models) {
    Write-Host "Downloading $($model.Name)..."

    if (Test-Path $model.Path) {
        Write-Host "  Already exists, skipping."
        continue
    }

    try {
        Invoke-WebRequest -Uri $model.URL -OutFile $model.Path -UseBasicParsing
        Write-Host "  ✓ Downloaded successfully"
    } catch {
        Write-Host "  ✗ Failed: $_"
    }
}

Write-Host "`nModel downloads complete!"
Write-Host "Note: Realistic Vision V5.1 must be downloaded manually from Civitai (requires account)"

## Step 4: Run Automation CLI (Optional Dry Run)

Once the workflow is exported to `workflows\sora-removal-production.json`, you
can trigger the pipeline automation directly from this repo:

```powershell
set PYTHONPATH $PWD\src
python -m pipeline.cli input\sora-test.mp4 `
  --workflow workflows\sora-removal-production.json `
  --host http://localhost:8188 `
  --output-dir output `
  --logs-dir logs
```

Add `--keep-temp` to inspect intermediate frames or `--overwrite` to replace the
input file (a `.bak` copy is kept unless `--no-backup` is provided).
```

## Step 4: Get sirioberati Workflow

```powershell
# Clone sirioberati's repository to get workflow JSON
cd C:\sora-watermark-saas
git clone https://github.com/sirioberati/content-aware-inpainting-v2.git

# The workflow JSON should be in the repo
# Copy to ComfyUI workflows directory
Copy-Item "content-aware-inpainting-v2\inpainting-workflow.json" -Destination "ComfyUI\ComfyUI\workflows\sora-removal.json"
```

## Step 5: Verify Installation

```powershell
# Restart ComfyUI
cd C:\sora-watermark-saas\ComfyUI
.\run_nvidia_gpu.bat

# In browser (http://127.0.0.1:8188):
# 1. Click "Load" button
# 2. Select workflows/sora-removal.json
# 3. You should see the full workflow with nodes:
#    - Video Input
#    - SAM 2 Segmentation
#    - DiffuEraser Inpainting
#    - Video Output

# If any nodes are red/missing:
# - Check custom_nodes installation
# - Check model paths in node settings
```

## Troubleshooting

### "CUDA out of memory"
```powershell
# Check VRAM usage
nvidia-smi

# Reduce batch size in DiffuEraser node
# Or restart ComfyUI to clear VRAM
```

### "Module not found" errors
```powershell
# Reinstall custom node dependencies
cd C:\sora-watermark-saas\ComfyUI\ComfyUI\custom_nodes\[NODE_NAME]
..\..\python_embeded\python.exe -m pip install -r requirements.txt --force-reinstall
```

### Models not loading
```
# Verify model paths match:
C:\sora-watermark-saas\ComfyUI\ComfyUI\models\checkpoints\pcm_sd15_smallcfg_2step_converted.safetensors
C:\sora-watermark-saas\ComfyUI\ComfyUI\models\sam2\sam2_hiera_base_plus.pt
```

## Next Steps

After successful installation:
1. Test workflow with a sample Sora video
2. Tune SAM 2 point prompts for logo detection
3. Adjust DiffuEraser parameters for quality
4. Move to audio preservation pipeline (Day 3)

## Expected Installation Time

- ComfyUI download + extract: 15-20 min
- Custom nodes install: 20-30 min
- Model downloads: 30-60 min (depending on internet)
- **Total: ~90 minutes**
