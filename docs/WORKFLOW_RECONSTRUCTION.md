# Workflow Reconstruction Guide

**⚠️ UPDATE (2025-10-06)**: sirioberati repository was FOUND at new URL!
**New Location**: https://github.com/sirioberati/content-aware-inpainting-v2
**Status**: This document is now a **BACKUP/REFERENCE** guide

**Use this document IF**:
- sirioberati's workflow doesn't work for your use case
- You want to customize the workflow significantly
- You need to understand workflow architecture for debugging

**Primary approach**: Use sirioberati's `inpainting-workflow.json` from the repo above

---

## Original Documentation (for reference)

---

## What We Know From sirioberati's Approach

### Architecture
```
Video Input
    ↓
SAM 2 Video Segmentation (track logo across frames)
    ↓
Binary Masks (per-frame masks for watermark region)
    ↓
DiffuEraser Inpainting (temporal-consistent removal)
    ↓
Video Output (with audio preserved separately)
```

### Components Used
1. **ComfyUI-VideoHelperSuite**: Load video, extract frames
2. **ComfyUI-SAM2**: Segment logo using point prompts
3. **ComfyUI_DiffuEraser**: Inpaint masked regions with temporal consistency
4. **Output nodes**: Save processed video/frames

---

## Manual Workflow Creation (ComfyUI GUI)

### Step 1: Video Input (VideoHelperSuite)

**Node**: `Load Video (VHS)`
- **Input**: Path to video file
- **Output**: Image batch (all frames)

**Settings**:
- Frame rate: Auto-detect
- Start frame: 0
- Frame count: -1 (all frames)

---

### Step 2: SAM 2 Segmentation

**Node**: `SAM2VideoSegmentation`
- **Inputs**:
  - Image batch (from Load Video)
  - Point coordinates (X, Y) - user clicks logo center
  - Point labels: [1] (foreground)

**Settings**:
- Model: `sam2_hiera_base_plus.pt`
- Track across frames: Enabled
- Propagation mode: Bidirectional

**Output**: Binary masks (per-frame)

**Critical Parameters**:
```
point_coords: [[960, 540]]  # Example for 1080p center
point_labels: [1]           # 1 = foreground (logo)
track_forward: True
track_backward: True
```

---

### Step 3: Mask Refinement (Optional)

**Node**: `MaskBlur` or `MaskDilate`
- **Purpose**: Soften mask edges to reduce artifacts
- **Settings**:
  - Blur radius: 3-5 pixels
  - Feather: 2-3 pixels

---

### Step 4: DiffuEraser Inpainting

**Node**: `DiffuEraser`
- **Inputs**:
  - Image batch (original frames)
  - Mask batch (from SAM 2)

**Settings**:
- Checkpoint: `pcm_sd15_smallcfg_2step_converted.safetensors`
- Steps: 10-25 (higher = better quality, slower)
- Temporal consistency weight: 0.7-0.9
- Guidance scale: 7.5

**Critical for temporal consistency**:
```
steps: 15                    # Start here, tune up if needed
temporal_weight: 0.8         # High = less flicker
cfg_scale: 7.5               # Guidance strength
seed: 42                     # Fixed seed for consistency
```

---

### Step 5: Video Output

**Node**: `Video Combine (VHS)`
- **Inputs**: Processed frames
- **Settings**:
  - FPS: Match input video
  - Codec: H.264
  - CRF: 18 (high quality)
  - Audio: None (handled separately by audio_preservation.py)

---

## Complete Node Graph (Minimal)

```
[Load Video (VHS)]
    ↓ image_batch
[SAM2VideoSegmentation]
    ↓ masks
[DiffuEraser]
    ↓ processed_frames
[Video Combine (VHS)]
    ↓ output.mp4
```

---

## Alternative: Community Workflows

Since sirioberati is down, check these alternatives:

### 1. Search ComfyUI Workflows Database
- https://comfyworkflows.com
- Search: "watermark removal" OR "inpainting video"
- Filter: Uses SAM2 + inpainting nodes

### 2. DiffuEraser Examples
- https://github.com/ShmuelRonen/ComfyUI_DiffuEraser/tree/main/workflows
- Check if repo has example workflows
- May include video inpainting examples

### 3. SAM 2 Video Examples
- https://github.com/kijai/ComfyUI-SAM2/tree/main/workflows
- Video segmentation examples
- Combine with DiffuEraser manually

---

## Week 1 Revised Timeline

### Day 1-2: Setup (UNCHANGED)
- Install ComfyUI + custom nodes
- Download models
- Verify installation

### Day 3: Workflow Creation (NEW - CRITICAL)

**Tasks**:
1. Create basic workflow in ComfyUI GUI
   - Load Video → SAM 2 → DiffuEraser → Save Video
   - Test on 10-second video clip
   - Verify all nodes connect without errors

2. Test SAM 2 point prompts
   - Click different positions on logo
   - Verify masks capture entire watermark
   - Test tracking across frames (play mask preview)

3. Tune DiffuEraser parameters
   - Start: steps=10, temporal_weight=0.8
   - Process test clip
   - Inspect for flicker, artifacts, blur
   - Increase steps if quality poor (max 25)

4. Document optimal settings
   - Save working workflow as `sora-removal-v1.json`
   - Note all parameter values
   - Screenshot node graph

**Success Criteria**:
- [ ] Workflow processes 10s test video without errors
- [ ] SAM 2 masks track logo accurately (>90% coverage)
- [ ] DiffuEraser output has minimal flicker
- [ ] Processing time < 2 minutes for 10s @ 1080p

**Estimated Time**: 4-6 hours (trial and error)

---

### Day 4: Audio Integration

**Tasks**:
1. Test audio preservation pipeline
   - Extract frames from test video
   - Run through ComfyUI workflow (frames only)
   - Stitch with audio using `audio_preservation.py`
   - Validate sync <33ms

2. End-to-End Test
   - Full pipeline: upload → process → stitch → download
   - Verify audio intact and synced
   - Check output quality

**Success Criteria**:
- [ ] Audio preservation working
- [ ] Sync validation passes
- [ ] End-to-end test completes

**Estimated Time**: 3-4 hours

---

### Day 5: Quality Validation (UNCHANGED)

- Process 10 diverse test videos
- Inspect frame-by-frame for artifacts
- Lock optimal parameters
- Export production workflow

---

## Troubleshooting Workflow Creation

### Issue: SAM 2 masks don't track logo
**Solutions**:
- Try multiple point prompts (3-5 points across logo)
- Use `SAM2AutoSegmentation` instead (no points, auto-detect)
- Reduce video to 720p (easier tracking)

### Issue: DiffuEraser produces flicker
**Solutions**:
- Increase temporal consistency weight (0.9)
- Increase steps (20-25)
- Enable mask feathering/blur

### Issue: Processing too slow
**Solutions**:
- Reduce video resolution (1080p → 720p)
- Lower DiffuEraser steps (10)
- Process shorter clips (10-15 seconds max)

### Issue: Output quality poor
**Solutions**:
- Increase DiffuEraser steps (25)
- Try different checkpoint (Realistic Vision V5.1)
- Adjust guidance scale (5.0-10.0)

---

## Fallback Plan

If workflow creation takes >2 days:

### Option A: Use Per-Frame Inpainting
- Skip DiffuEraser (temporal consistency)
- Use standard SD Inpainting node
- Post-process with temporal smoothing filter
- **Tradeoff**: May have more flicker

### Option B: Delay Week 2
- Extend Week 1 to 7-10 days
- Get workflow perfect before automation
- **Tradeoff**: Delays MVP launch

### Option C: Hire ComfyUI Expert (Fiverr/Upwork)
- Post job: "Create ComfyUI workflow for video watermark removal"
- Provide: SAM 2 + DiffuEraser requirements
- Budget: $100-300 for working workflow
- **Tradeoff**: Cost, but saves 3-5 days

---

## Resources for Workflow Creation

### ComfyUI Tutorials
- https://www.youtube.com/results?search_query=comfyui+video+inpainting
- https://www.youtube.com/results?search_query=comfyui+sam2

### Community Discord
- ComfyUI Discord: https://discord.gg/comfyui
- Ask: "Video watermark removal workflow with SAM2 + DiffuEraser"

### Node Documentation
- SAM 2: https://github.com/kijai/ComfyUI-SAM2#usage
- DiffuEraser: https://github.com/ShmuelRonen/ComfyUI_DiffuEraser#examples

---

## Updated Week 1 Checklist

- [ ] Day 1-2: ComfyUI setup (UNCHANGED)
- [ ] **Day 3: Create workflow from scratch (NEW)**
- [ ] **Day 4: Audio integration (MOVED)**
- [ ] Day 5: Quality validation (UNCHANGED)

**New Estimated Time**: 6-8 days (was 5 days)

**Status**: Reconstruct workflow manually using known components
**Risk Level**: Medium (doable but requires experimentation)

---

**Last Updated**: 2025-10-06 (after sirioberati repo went offline)
