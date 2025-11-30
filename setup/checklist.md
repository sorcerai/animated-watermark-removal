# Week 1 Setup Checklist

Track your progress through the foundation phase.

## Day 1-2: ComfyUI Installation

### Prerequisites
- [ ] Windows 3090 box accessible
- [ ] NVIDIA driver 5xx.xx+ installed
- [ ] `nvidia-smi` shows RTX 3090 with 24GB VRAM
- [ ] Git installed
- [ ] 50GB+ free disk space

### ComfyUI Portable
- [ ] Downloaded ComfyUI portable (~4GB)
- [ ] Extracted to `C:\sora-watermark-saas\ComfyUI`
- [ ] Ran `run_nvidia_gpu.bat` successfully
- [ ] Browser opens to http://localhost:8188
- [ ] Can see ComfyUI interface

### Custom Nodes (6 total)
- [ ] DiffuEraser installed + dependencies
- [ ] VideoHelperSuite installed + dependencies
- [ ] SAM2 installed + dependencies
- [ ] KJNodes installed + dependencies
- [ ] LayerStyle installed + dependencies
- [ ] Easy-Use installed + dependencies

### Models Downloaded
- [ ] DiffuEraser checkpoint (~1.8GB)
- [ ] Realistic Vision V5.1 base model (~2GB)
- [ ] SAM 2 Base model (~400MB)
- [ ] All models in correct directories
- [ ] ComfyUI recognizes models (no red nodes)

### Workflow Import
- [ ] Cloned sirioberati repo
- [ ] Copied workflow JSON to ComfyUI/workflows/
- [ ] Loaded workflow in ComfyUI GUI
- [ ] All nodes load without errors

**Estimated time**: 90-120 minutes
**Success criteria**: Can load workflow in GUI, all nodes green

---

## Day 3: Audio Preservation

### Setup
- [ ] Python 3.8+ available
- [ ] FFmpeg installed (via winget or chocolatey)
- [ ] `ffmpeg -version` works in PowerShell
- [ ] Copied `audio_preservation.py` to project
- [ ] Have test video (10-30 seconds, MP4 with audio)

### Testing
- [ ] Ran `test-pipeline.py` with test video
- [ ] All 4 tests passed:
  - [ ] ✓ Connection (ComfyUI running)
  - [ ] ✓ Models (can access info)
  - [ ] ✓ Audio (preservation pipeline works)
  - [ ] ✓ Workflow (JSON loads)
- [ ] Output video has audio
- [ ] Audio sync < 33ms difference

**Estimated time**: 2-3 hours
**Success criteria**: `test-pipeline.py` all green

---

## Day 4-5: Quality Validation

### Test Video Collection
- [ ] Have 10 diverse Sora videos:
  - [ ] 3 with textured backgrounds
  - [ ] 3 with smooth backgrounds
  - [ ] 2 with camera motion
  - [ ] 2 with objects moving behind logo
- [ ] All videos 10-30 seconds
- [ ] All have audio tracks

### ComfyUI Processing (Manual)
- [ ] Loaded workflow in GUI
- [ ] Processed video #1
  - [ ] Logo detected by SAM 2?
  - [ ] Watermark removed cleanly?
  - [ ] Audio intact?
  - [ ] Notes: ___________________
- [ ] Processed video #2
  - [ ] Notes: ___________________
- [ ] Processed video #3
  - [ ] Notes: ___________________
- [ ] Processed videos #4-10
  - [ ] Success rate: ___/10

### Parameter Tuning
- [ ] SAM 2 point prompt positions documented
- [ ] DiffuEraser steps optimized
- [ ] Temporal consistency weight tuned
- [ ] Output quality/codec locked

### Quality Inspection (Frame-by-Frame)
- [ ] No temporal flicker (play at 1x speed)
- [ ] No obvious blur/smearing in logo area
- [ ] No ghosting (faint watermark remnants)
- [ ] Background texture matches surroundings
- [ ] Audio sync perfect (<1 frame)

### Production Workflow Export
- [ ] Saved optimal workflow as `sora-removal-production.json`
- [ ] Documented all parameter values
- [ ] Created quick reference guide
- [ ] Ready to automate

**Estimated time**: 1-2 days
**Success criteria**: 8/10 videos process successfully with acceptable quality

---

## Blockers & Notes

### Issues Encountered
```
Date: ___________
Issue: _______________________________________
Solution: _____________________________________
Time lost: _____ hours

Date: ___________
Issue: _______________________________________
Solution: _____________________________________
Time lost: _____ hours
```

### Quality Issues
```
Video: ____________
Problem: [ ] Flicker [ ] Blur [ ] Audio [ ] Detection [ ] Other: _______
Severity: [ ] Minor [ ] Moderate [ ] Critical
Fixed? [ ] Yes [ ] No

Video: ____________
Problem: [ ] Flicker [ ] Blur [ ] Audio [ ] Detection [ ] Other: _______
Severity: [ ] Minor [ ] Moderate [ ] Critical
Fixed? [ ] Yes [ ] No
```

### Performance Notes
```
Average processing time (10s @ 1080p): _____ seconds
GPU utilization: _____%
VRAM usage: _____ GB / 24 GB
Bottleneck: [ ] GPU [ ] CPU [ ] Disk I/O [ ] None
```

---

## Week 1 Completion Criteria

Before moving to Week 2 (Automation):

✅ **Must Have**:
- [ ] ComfyUI running stable on 3090
- [ ] Can process videos with audio preservation
- [ ] 80%+ success rate on test videos
- [ ] Quality acceptable for MVP

⚠️ **Nice to Have**:
- [ ] Processing time < 30s for 10s video
- [ ] Zero manual intervention needed
- [ ] 95%+ success rate

🚫 **Blockers** (stop and fix before Week 2):
- [ ] GPU crashes/OOM errors
- [ ] Audio consistently out of sync
- [ ] <50% videos process successfully
- [ ] Critical quality issues (heavy artifacts)

---

## Ready for Week 2?

If you've checked all "Must Have" boxes and have no critical blockers:

✅ **YES** - Proceed to automation (ComfyUI API wrapper, queue system)

⚠️ **NOT YET** - Resolve blockers, improve success rate, stabilize quality

---

**Time Spent on Week 1**: _____ hours
**Original Estimate**: 40 hours (1 week)
**Variance**: _____ hours (over/under)

**Lessons Learned**:
1. _____________________________________
2. _____________________________________
3. _____________________________________
