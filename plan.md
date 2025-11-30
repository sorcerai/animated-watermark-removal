# Sora Watermark Remover - Development Plan

**Full Plan**: See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for complete 4-week execution strategy

## Current Phase: Week 1 - Foundation

**Status**: Setup phase
**Goal**: ComfyUI operational + workflow validated + audio preservation tested
**Timeline**: Days 1-8 (originally 5 days, extended for workflow creation)

### Week 1 Checklist

**Day 1-2: ComfyUI Installation**
- [ ] Download ComfyUI Portable for Windows (~4GB)
- [ ] Install 6 custom nodes (DiffuEraser, SAM2, VideoHelper, KJNodes, LayerStyle, Easy-Use)
- [ ] Download 3 models: DiffuEraser checkpoint, SAM 2, Realistic Vision (~4GB total)
- [ ] Clone sirioberati workflow from https://github.com/sirioberati/content-aware-inpainting-v2
- [ ] Verify installation: ComfyUI loads, all nodes visible, workflow imports

**Day 3: Workflow Testing** (4-6 hours)
- [ ] Load sirioberati's `inpainting-workflow.json` in ComfyUI GUI
- [ ] Prepare 10-second test clip from Sora video
- [ ] Test SAM 2 point prompts for logo detection (verify >90% tracking)
- [ ] Tune DiffuEraser parameters (steps: 15-25, temporal_weight: 0.8-0.9)
- [ ] Document optimal settings, save as `sora-removal-v1.json`

**Day 4: Audio Integration** (3-4 hours)
- [ ] Test `audio_preservation.py` pipeline
- [ ] Verify FFmpeg `-c:a copy` for bit-perfect audio
- [ ] End-to-end test: extract frames → process → stitch → validate
- [ ] Audio sync validation <33ms tolerance

**Day 5: Quality Validation** (1-2 days)
- [ ] Process 10 diverse test videos (1080p 30fps, 60fps, 720p, 4K→1080p)
- [ ] Frame-by-frame inspection (flicker, blur, ghosting, audio sync)
- [ ] Lock production parameters in `docs/PRODUCTION_PARAMS.md`
- [ ] Export `workflows/sora-removal-production.json`
- [ ] Commit to git

### Week 1 Success Criteria
- ✅ ComfyUI running on Windows 3090 with all dependencies
- ✅ sirioberati workflow processing videos successfully
- ✅ Audio preservation validated (<33ms sync, bit-perfect copy)
- ✅ 8/10 test videos pass quality checklist (80% success rate)
- ✅ Production parameters locked and documented

---

## Future Phases (Overview)

### Week 2: Automation (Days 6-12)
- **Goal**: API + job queue + storage layer
- **Key Tasks**:
  - ComfyUI API client (Python + Node.js wrapper)
  - Redis + BullMQ job queue setup
  - Cloudflare R2 storage integration
  - Presigned upload/download URLs
  - Job status polling endpoints

### Week 3: Backend Complete (Days 13-19)
- **Goal**: Full backend SaaS infrastructure
- **Key Tasks**:
  - API routes (upload, status, download)
  - Clerk authentication
  - Stripe billing integration
  - RunPod serverless overflow
  - Rate limiting + error handling
  - Production deployment

### Week 4: Frontend + Launch (Days 20-28)
- **Goal**: User-facing interface + beta launch
- **Key Tasks**:
  - Next.js upload/download UI
  - Progress indicators + real-time updates
  - Vercel deployment
  - Beta user testing (10 users)
  - Performance monitoring + refinement
  - Public launch preparation

---

## Risk Mitigation

**Week 1 Risks**:
- sirioberati repo offline → **Mitigated**: Found renamed repo at new URL, created `WORKFLOW_RECONSTRUCTION.md` as backup
- Workflow creation complexity → **Buffer**: Extended timeline from 5 to 6-8 days
- GPU compatibility issues → **Mitigation**: Comprehensive troubleshooting guide in `windows-3090-setup.md`

**Overall Risks** (see MASTER_PLAN.md):
- Model licensing (DiffuEraser non-commercial) → Contact authors for commercial license
- 3090 capacity constraints → RunPod serverless overflow ready
- Audio sync issues → Validated pipeline with <33ms tolerance requirement

---

## Quick Commands

**Setup**: Follow `setup/windows/windows-3090-setup.md`
**Workflow**: Reference `docs/WORKFLOW_RECONSTRUCTION.md` if sirioberati repo unavailable
**Audio**: Use `src/core/audio_preservation.py` pipeline
**Quality**: Apply checklist from `week-1-setup.mdc` aidd rule

**Current Action**: Execute Week 1, Day 1-2 installation tasks

---

## Architecture Reference

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- System architecture diagrams
- Data flow patterns
- Component breakdown
- Technology stack decisions
- GPU scaling strategy

---

## Repository Structure

```
sora-watermark-remover/
├── docs/
│   ├── MASTER_PLAN.md              # Complete 4-week plan
│   ├── ARCHITECTURE.md             # System design
│   └── WORKFLOW_RECONSTRUCTION.md  # Backup workflow guide
├── setup/windows/
│   └── windows-3090-setup.md       # Installation guide
├── src/core/
│   └── audio_preservation.py       # FFmpeg audio pipeline
├── workflows/templates/
│   └── sora-removal-production.json # Locked production workflow
├── .aidd/rules/
│   ├── video-processing.mdc        # Video workflow rules
│   ├── saas-backend.mdc            # Backend development rules
│   └── week-1-setup.mdc            # Week 1 guidance
└── plan.md                         # This file (aidd /plan)
```

---

**Last Updated**: 2025-10-09
**Phase**: Week 1 - Foundation
**Next Review**: After Day 5 quality validation complete
