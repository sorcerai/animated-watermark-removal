# Sora Watermark Removal SaaS - Master Plan

**Timeline**: 4 weeks to MVP launch
**Hardware**: 3070 8GB + 3090 24GB (Windows)
**Approach**: Hybrid (sirioberati ComfyUI workflow + custom SaaS automation)
**Commercial Licensing**: Deferred for now

---

## Week 1: Foundation & Validation

**Goal**: ComfyUI running stable on 3090, audio preservation working, 80%+ success rate on test videos

### Day 1-2: ComfyUI Installation (90-120 minutes)

**Tasks**:
1. Install ComfyUI Portable for Windows
   - Download from GitHub releases (~4GB)
   - Extract to `C:\sora-watermark-saas\ComfyUI`
   - Run `run_nvidia_gpu.bat` → verify http://localhost:8188 loads

2. Install 6 Custom Nodes
   ```powershell
   cd C:\sora-watermark-saas\ComfyUI\ComfyUI\custom_nodes

   # DiffuEraser (core inpainting)
   git clone https://github.com/ShmuelRonen/ComfyUI_DiffuEraser.git
   cd ComfyUI_DiffuEraser
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..

   # SAM 2 (segmentation - CRITICAL)
   git clone https://github.com/kijai/ComfyUI-SAM2.git
   cd ComfyUI-SAM2
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..

   # VideoHelperSuite
   git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
   cd ComfyUI-VideoHelperSuite
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..

   # KJNodes (utilities)
   git clone https://github.com/kijai/ComfyUI-KJNodes.git
   cd ComfyUI-KJNodes
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..

   # LayerStyle (masking)
   git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
   cd ComfyUI_LayerStyle
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..

   # Easy-Use (workflow helpers)
   git clone https://github.com/yolain/ComfyUI-Easy-Use.git
   cd ComfyUI-Easy-Use
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   cd ..
   ```

3. Download Models (4GB total)
   - DiffuEraser checkpoint: `pcm_sd15_smallcfg_2step_converted.safetensors` (~1.8GB)
   - Realistic Vision V5.1: `realisticVisionV51_v51VAE.safetensors` (~2GB)
   - SAM 2 Base: `sam2_hiera_base_plus.pt` (~400MB)
   - Place in correct `models/` subdirectories

4. Import sirioberati Workflow
   ```powershell
   cd C:\sora-watermark-saas
   git clone https://github.com/sirioberati/content-aware-inpainting-v2.git
   Copy-Item "content-aware-inpainting-v2\inpainting-workflow.json" `
     -Destination "ComfyUI\ComfyUI\workflows\sora-removal.json"
   ```

**Success Criteria**:
- [ ] ComfyUI GUI loads without errors
- [ ] All custom nodes appear in node menu
- [ ] All models show in checkpoint/SAM2 dropdowns (no red nodes)
- [ ] Workflow JSON loads successfully

**Estimated Time**: 90-120 minutes

---

### Day 3: Audio Preservation Pipeline (2-3 hours)

**Tasks**:
1. Install FFmpeg on Windows
   ```powershell
   winget install Gyan.FFmpeg
   # Verify
   ffmpeg -version
   ```

2. Test Audio Preservation Script
   - Copy test video (10-30s, MP4 with audio) to project root
   - Run `python tests/test-pipeline.py test-video.mp4`
   - Expected output:
     ```
     ✓ ComfyUI is running
     ✓ Can access model info
     ✓ Audio preservation pipeline working
     ✓ Workflow loaded
     ```

3. Validate Audio Sync
   - Duration difference < 33ms (1 frame @ 30fps)
   - Audio track present in output
   - No pops/clicks/artifacts

**Success Criteria**:
- [ ] `test-pipeline.py` all 4 tests pass
- [ ] Output video has audio
- [ ] Audio sync validated < 33ms difference

**Estimated Time**: 2-3 hours

---

### Day 4-5: Quality Validation (1-2 days)

**Tasks**:
1. Collect 10 Diverse Sora Videos
   - 3 with textured backgrounds (grass, brick, fabric)
   - 3 with smooth backgrounds (sky, walls, water)
   - 2 with camera motion (pan, zoom, dolly)
   - 2 with objects moving behind logo
   - All 10-30 seconds with audio

2. Process Through ComfyUI GUI (Manual)
   - Load workflow: `sora-removal.json`
   - For each video:
     - Upload video
     - Set SAM 2 point prompts (click logo center)
     - Adjust DiffuEraser steps if needed
     - Queue → Wait → Download result
   - Document success/failure for each

3. Quality Inspection (Frame-by-Frame)
   - Play at 1x speed, watch for:
     - ❌ Temporal flicker
     - ❌ Blur/smearing in logo area
     - ❌ Ghosting (faint watermark remnants)
     - ✅ Background texture matches surroundings
     - ✅ Audio sync perfect (<1 frame)

4. Parameter Tuning
   - SAM 2 point prompt positions
   - DiffuEraser inference steps (default 10, try 15-25)
   - Temporal consistency weight
   - Output codec/quality settings

5. Lock Production Workflow
   - Save optimal parameters as `sora-removal-production.json`
   - Document all settings in `setup/optimal-parameters.md`
   - Create quick reference guide for SAM 2 prompting

**Success Criteria**:
- [ ] 8/10 videos process successfully
- [ ] No critical quality issues (heavy artifacts, lost audio)
- [ ] Production workflow exported
- [ ] Average processing time < 60s for 10s @ 1080p video

**Estimated Time**: 1-2 days

---

### Week 1 Completion Checklist

✅ **Must Have** (block Week 2 if not met):
- [ ] ComfyUI running stable on 3090
- [ ] Can process videos with audio preservation
- [ ] 80%+ success rate on test videos
- [ ] Quality acceptable for MVP

⚠️ **Nice to Have**:
- [ ] Processing time < 30s for 10s video
- [ ] Zero manual intervention needed
- [ ] 95%+ success rate

🚫 **Blockers** (STOP and fix before Week 2):
- [ ] GPU crashes/OOM errors
- [ ] Audio consistently out of sync
- [ ] <50% videos process successfully
- [ ] Critical quality issues

---

## Week 2: API & Automation

**Goal**: ComfyUI API integrated, queue system working, video storage layer operational

### Day 1: ComfyUI API Client (6-8 hours)

**Tasks**:
1. Build API Wrapper (`src/comfyui/client.py`)
   - WebSocket connection for progress updates
   - File upload via `/upload/image` endpoint
   - Workflow queue via `/prompt` endpoint
   - Status polling via `/history` and `/queue`
   - Result download from `/view` endpoint

2. Create High-Level Interface
   ```python
   class ComfyUIClient:
       def __init__(self, host="http://localhost:8188"):
           self.host = host
           self.ws = None

       def process_video(
           self,
           video_path: str,
           workflow_path: str,
           sam2_point: tuple[int, int],
           callback: Optional[Callable] = None
       ) -> dict:
           """
           Process video through ComfyUI workflow

           Returns:
               {
                   "prompt_id": str,
                   "status": "success" | "failed",
                   "output_frames": Path,
                   "processing_time": float
               }
           """
   ```

3. Test End-to-End
   - Upload test video
   - Queue workflow with SAM 2 point
   - Monitor WebSocket for progress
   - Download processed frames
   - Verify against manual GUI results

**Success Criteria**:
- [ ] Can queue workflows programmatically
- [ ] WebSocket progress updates working
- [ ] Matches GUI output quality
- [ ] Processing time within 10% of GUI

**Estimated Time**: 6-8 hours

---

### Day 2: Queue System (4-6 hours)

**Tasks**:
1. Install Redis + BullMQ
   ```bash
   # Windows: Install Redis via WSL2 or Docker
   docker run -d -p 6379:6379 redis:alpine

   # Node.js dependencies
   npm install bullmq ioredis
   ```

2. Create Job Queue (`src/api/queue/video-queue.ts`)
   ```typescript
   import { Queue, Worker } from 'bullmq';

   const videoQueue = new Queue('video-processing', {
       connection: { host: 'localhost', port: 6379 }
   });

   const worker = new Worker('video-processing', async (job) => {
       const { videoPath, userId, samPoint } = job.data;

       // Extract frames
       const frames = await extractFrames(videoPath);

       // Process through ComfyUI
       const result = await comfyClient.process_video(
           videoPath,
           'sora-removal-production.json',
           samPoint
       );

       // Stitch with audio
       const output = await stitchWithAudio(result.output_frames, videoPath);

       // Upload to storage
       const url = await uploadToR2(output, userId);

       return { outputUrl: url, status: 'completed' };
   });
   ```

3. Add Job Events
   - `onProgress`: Update user on % complete
   - `onCompleted`: Webhook notification
   - `onFailed`: Error logging + retry logic (max 2 retries)

**Success Criteria**:
- [ ] Jobs queue and process asynchronously
- [ ] Can handle 3 concurrent jobs (3090 limit)
- [ ] Failed jobs retry automatically
- [ ] Progress updates via WebSocket to frontend

**Estimated Time**: 4-6 hours

---

### Day 3: Storage Layer (3-4 hours)

**Tasks**:
1. Setup Cloudflare R2
   - Create R2 bucket: `sora-watermark-videos`
   - Generate API token with read/write permissions
   - Install SDK: `npm install @aws-sdk/client-s3`

2. Create Storage Client (`src/api/storage/r2.ts`)
   ```typescript
   import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
   import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

   class R2Storage {
       private client: S3Client;

       async uploadVideo(
           filePath: string,
           userId: string,
           videoId: string
       ): Promise<string> {
           const key = `${userId}/${videoId}/output.mp4`;
           const file = await fs.readFile(filePath);

           await this.client.send(new PutObjectCommand({
               Bucket: 'sora-watermark-videos',
               Key: key,
               Body: file,
               ContentType: 'video/mp4'
           }));

           return key;
       }

       async getDownloadUrl(key: string, expiresIn: number = 3600): Promise<string> {
           const command = new GetObjectCommand({
               Bucket: 'sora-watermark-videos',
               Key: key
           });
           return await getSignedUrl(this.client, command, { expiresIn });
       }
   }
   ```

3. Lifecycle Policies
   - Delete processed videos after 7 days (configurable)
   - Archive to cheaper storage after 24 hours

**Success Criteria**:
- [ ] Can upload processed videos to R2
- [ ] Can generate signed download URLs
- [ ] Lifecycle policies configured
- [ ] Average upload time < 5s for 10s video

**Estimated Time**: 3-4 hours

---

### Day 4-5: Integration Testing (8-12 hours)

**Tasks**:
1. End-to-End Test Script
   ```python
   # Test full pipeline
   job_id = queue.add_job({
       'video_url': 'test-video.mp4',
       'user_id': 'test-user',
       'sam_point': (960, 540)  # 1080p center
   })

   # Monitor progress
   while job.status != 'completed':
       print(f"Progress: {job.progress}%")
       time.sleep(1)

   # Download result
   result_url = job.result['outputUrl']
   download_video(result_url, 'output.mp4')

   # Validate
   assert audio_synced('test-video.mp4', 'output.mp4')
   assert quality_acceptable('output.mp4')
   ```

2. Load Testing
   - Queue 10 videos simultaneously
   - Verify queue throttles to 3 concurrent (3090 limit)
   - Measure throughput: ~120 videos/hour @ 10s each

3. Error Handling
   - Test OOM errors (submit 4K video)
   - Test corrupted videos
   - Test missing audio tracks
   - Verify retry logic and error messages

**Success Criteria**:
- [ ] Can process 10 videos end-to-end without manual intervention
- [ ] Queue correctly throttles concurrent jobs
- [ ] Error handling graceful (no crashes)
- [ ] All outputs pass audio sync validation

**Estimated Time**: 8-12 hours

---

### Week 2 Completion Checklist

✅ **Must Have**:
- [ ] ComfyUI API client working
- [ ] Queue system processing jobs asynchronously
- [ ] Storage layer operational
- [ ] End-to-end test passing

⚠️ **Nice to Have**:
- [ ] WebSocket progress updates to frontend
- [ ] Automatic retry on transient failures
- [ ] Video cleanup after 7 days

---

## Week 3: SaaS Backend

**Goal**: API routes live, authentication working, billing integrated, RunPod fallback configured

### Day 1-2: API Routes (8-12 hours)

**Tasks**:
1. Setup Express/Fastify Backend
   ```bash
   npm init -y
   npm install fastify @fastify/cors @fastify/multipart dotenv
   ```

2. Create Upload Endpoint (`src/api/routes/upload.ts`)
   ```typescript
   app.post('/api/upload', async (req, res) => {
       const data = await req.file();
       const { samPoint } = req.body;

       // Validate video format
       if (!['video/mp4', 'video/quicktime'].includes(data.mimetype)) {
           return res.code(400).send({ error: 'Invalid format' });
       }

       // Save to temp storage
       const tempPath = await saveTempFile(data);

       // Queue job
       const job = await videoQueue.add('process', {
           videoPath: tempPath,
           userId: req.user.id,
           samPoint: JSON.parse(samPoint)
       });

       return { jobId: job.id, status: 'queued' };
   });
   ```

3. Create Status Endpoint
   ```typescript
   app.get('/api/status/:jobId', async (req, res) => {
       const job = await videoQueue.getJob(req.params.jobId);

       if (!job) {
           return res.code(404).send({ error: 'Job not found' });
       }

       return {
           status: await job.getState(),
           progress: job.progress,
           result: await job.returnvalue
       };
   });
   ```

4. Create Download Endpoint
   ```typescript
   app.get('/api/download/:jobId', async (req, res) => {
       const job = await videoQueue.getJob(req.params.jobId);
       const state = await job.getState();

       if (state !== 'completed') {
           return res.code(400).send({ error: 'Not ready' });
       }

       const { outputUrl } = job.returnvalue;
       const signedUrl = await r2.getDownloadUrl(outputUrl);

       return { downloadUrl: signedUrl };
   });
   ```

**Success Criteria**:
- [ ] Upload endpoint accepts MP4 files
- [ ] Status endpoint returns real-time progress
- [ ] Download endpoint generates signed URLs
- [ ] Error handling for all edge cases

**Estimated Time**: 8-12 hours

---

### Day 2-3: Authentication (4-6 hours)

**Tasks**:
1. Setup Clerk
   ```bash
   npm install @clerk/clerk-sdk-node
   ```
   - Create Clerk application
   - Configure allowed sign-in methods (email, Google)
   - Add webhook for user.created events

2. Protect API Routes
   ```typescript
   import { requireAuth } from '@clerk/clerk-sdk-node';

   app.addHook('onRequest', requireAuth());

   app.post('/api/upload', async (req, res) => {
       const userId = req.auth.userId;  // From Clerk
       // ... rest of upload logic
   });
   ```

3. User Quotas
   - Free tier: 3 videos/month
   - Pro tier: 100 videos/month
   - Enterprise: Unlimited

   ```typescript
   async function checkQuota(userId: string): Promise<boolean> {
       const user = await clerkClient.users.getUser(userId);
       const tier = user.publicMetadata.tier || 'free';
       const usageThisMonth = await getUsage(userId);

       const limits = { free: 3, pro: 100, enterprise: Infinity };
       return usageThisMonth < limits[tier];
   }
   ```

**Success Criteria**:
- [ ] Only authenticated users can upload
- [ ] Quota enforcement working
- [ ] User metadata tracked in Clerk

**Estimated Time**: 4-6 hours

---

### Day 4: Billing Integration (6-8 hours)

**Tasks**:
1. Setup Stripe
   ```bash
   npm install stripe
   ```
   - Create products: Pro ($29/mo), Enterprise (custom)
   - Setup webhooks for subscription events

2. Create Checkout Session
   ```typescript
   app.post('/api/checkout', async (req, res) => {
       const session = await stripe.checkout.sessions.create({
           customer_email: req.user.email,
           mode: 'subscription',
           line_items: [{
               price: 'price_pro_monthly',
               quantity: 1
           }],
           success_url: 'https://yourapp.com/success',
           cancel_url: 'https://yourapp.com/pricing'
       });

       return { checkoutUrl: session.url };
   });
   ```

3. Handle Webhooks
   ```typescript
   app.post('/api/stripe/webhook', async (req, res) => {
       const event = stripe.webhooks.constructEvent(
           req.rawBody,
           req.headers['stripe-signature'],
           process.env.STRIPE_WEBHOOK_SECRET
       );

       if (event.type === 'customer.subscription.created') {
           await clerkClient.users.updateUserMetadata(userId, {
               publicMetadata: { tier: 'pro' }
           });
       }
   });
   ```

**Success Criteria**:
- [ ] Checkout flow working
- [ ] Subscriptions update user tier
- [ ] Webhooks handled correctly

**Estimated Time**: 6-8 hours

---

### Day 5: RunPod Serverless Fallback (4-6 hours)

**Tasks**:
1. Create RunPod Template
   - Base image: `runpod/pytorch:3.10-2.0.0-117`
   - Install ComfyUI + custom nodes
   - Pre-download all models
   - Expose handler endpoint

2. Create Handler (`deploy/runpod/handler.py`)
   ```python
   import runpod
   from comfyui_client import ComfyUIClient

   def process_video(job):
       video_url = job['input']['video_url']
       sam_point = job['input']['sam_point']

       # Download video
       video_path = download_from_url(video_url)

       # Process
       client = ComfyUIClient()
       result = client.process_video(video_path, sam_point)

       # Upload result
       output_url = upload_to_r2(result['output'])

       return {"output_url": output_url}

   runpod.serverless.start({"handler": process_video})
   ```

3. Queue Router Logic
   ```typescript
   // In video queue worker
   const queueLength = await videoQueue.count();
   const use3090 = queueLength <= 3;  // 3090 capacity

   if (use3090) {
       // Process locally on 3090
       await comfyClient.process_video(...);
   } else {
       // Overflow to RunPod
       const result = await runpodClient.run({
           video_url: await uploadTemp(videoPath),
           sam_point: samPoint
       });
   }
   ```

**Success Criteria**:
- [ ] RunPod template working
- [ ] Queue automatically routes overflow
- [ ] Cost per video < $0.01

**Estimated Time**: 4-6 hours

---

### Week 3 Completion Checklist

✅ **Must Have**:
- [ ] API routes functional
- [ ] Authentication working
- [ ] Billing integrated
- [ ] RunPod fallback configured

⚠️ **Nice to Have**:
- [ ] Webhook notifications to users
- [ ] Usage analytics dashboard
- [ ] Cost monitoring

---

## Week 4: Frontend & Launch

**Goal**: MVP frontend live, beta testing complete, production monitoring operational

### Day 1-2: Frontend (12-16 hours)

**Tasks**:
1. Setup Next.js Project
   ```bash
   npx create-next-app@latest sora-watermark-frontend
   cd sora-watermark-frontend
   npm install @clerk/nextjs zustand react-dropzone
   ```

2. Create Upload Page (`app/upload/page.tsx`)
   ```tsx
   'use client';
   import { useDropzone } from 'react-dropzone';
   import { useState } from 'react';

   export default function Upload() {
       const [jobId, setJobId] = useState<string | null>(null);
       const [progress, setProgress] = useState(0);

       const { getRootProps, getInputProps } = useDropzone({
           accept: {'video/mp4': ['.mp4'], 'video/quicktime': ['.mov']},
           maxSize: 500 * 1024 * 1024,  // 500MB
           onDrop: async (files) => {
               const formData = new FormData();
               formData.append('file', files[0]);
               formData.append('samPoint', JSON.stringify([960, 540]));

               const res = await fetch('/api/upload', {
                   method: 'POST',
                   body: formData
               });

               const { jobId } = await res.json();
               setJobId(jobId);
               pollStatus(jobId);
           }
       });

       async function pollStatus(id: string) {
           const interval = setInterval(async () => {
               const res = await fetch(`/api/status/${id}`);
               const { status, progress } = await res.json();

               setProgress(progress);

               if (status === 'completed') {
                   clearInterval(interval);
                   // Show download button
               }
           }, 1000);
       }

       return (
           <div {...getRootProps()}>
               <input {...getInputProps()} />
               <p>Drop video here or click to upload</p>
               {jobId && <ProgressBar value={progress} />}
           </div>
       );
   }
   ```

3. Add SAM Point Selector (Interactive)
   - Show first frame thumbnail
   - Let user click logo center
   - Pass coordinates to API

4. Create Download Page
   - Show before/after preview
   - Download button with signed URL
   - Share link generation

**Success Criteria**:
- [ ] Upload flow working end-to-end
- [ ] Real-time progress updates
- [ ] Download works
- [ ] Mobile responsive

**Estimated Time**: 12-16 hours

---

### Day 3: Testing & Refinement (6-8 hours)

**Tasks**:
1. Beta Testing
   - Invite 10 users
   - Process 50+ real Sora videos
   - Collect feedback on quality, speed, UX

2. Bug Fixes
   - Fix edge cases discovered in beta
   - Improve error messages
   - Optimize upload speed

3. Documentation
   - User guide (how to select SAM point)
   - FAQ (common issues)
   - API docs for enterprise customers

**Success Criteria**:
- [ ] 90%+ success rate on beta videos
- [ ] No critical bugs
- [ ] User guide complete

**Estimated Time**: 6-8 hours

---

### Day 4: Production Deployment (4-6 hours)

**Tasks**:
1. Deploy Backend
   - Host API on Railway/Render
   - Redis on Upstash
   - ComfyUI on dedicated Windows 3090 box

2. Deploy Frontend
   - Vercel deployment
   - Custom domain setup
   - SSL certificates

3. Monitoring
   - Sentry for error tracking
   - LogRocket for session replay
   - Plausible for analytics

**Success Criteria**:
- [ ] All services live
- [ ] SSL working
- [ ] Monitoring operational

**Estimated Time**: 4-6 hours

---

### Day 5: Launch & Iteration (Variable)

**Tasks**:
1. Soft Launch
   - Post on Twitter, Reddit (r/StableDiffusion, r/SideProject)
   - Share in AI communities
   - Launch on Product Hunt (optional)

2. Monitor First 100 Users
   - Track success rate
   - Identify bottlenecks
   - Collect feedback

3. Rapid Iteration
   - Fix critical issues within 24 hours
   - Deploy improvements based on feedback

**Success Criteria**:
- [ ] 100+ users signed up
- [ ] >85% success rate
- [ ] No major outages

---

## Week 4 Completion Checklist

✅ **Launch Ready**:
- [ ] Frontend deployed
- [ ] Backend stable
- [ ] Monitoring operational
- [ ] 100+ beta users processed videos

---

## Risk Mitigation Strategies

### Technical Risks

**GPU OOM on 4K Videos**
- Mitigation: Downscale to 1080p for processing, upscale output
- Fallback: Reject 4K uploads until post-MVP

**Poor Detection on Edge Cases**
- Mitigation: Manual SAM point selection UI
- Fallback: Offer "manual mask upload" option

**Temporal Flicker**
- Mitigation: Tune DiffuEraser temporal consistency weight
- Fallback: Add post-processing smoothing filter

**Audio Desync**
- Mitigation: Strict <33ms validation, reject if fails
- Fallback: Offer "no audio" output option

### Business Risks

**Commercial Licensing**
- Mitigation: Contact ProPainter/DiffuEraser authors
- Fallback: Switch to SAM 2 + Stable Diffusion Inpainting (Apache 2.0)

**GPU Cost Overruns**
- Mitigation: Aggressive queue throttling, use 3090 first
- Fallback: Increase pricing, add processing time tiers

**Low Conversion Rate**
- Mitigation: Free tier (3 videos/month) to drive adoption
- Fallback: Partner with content creators for bulk licensing

---

## KPIs & Success Metrics

### Week 1
- [ ] ComfyUI setup time < 2 hours
- [ ] 80%+ success rate on 10 test videos
- [ ] Audio sync <33ms on all outputs

### Week 2
- [ ] API processes 100 videos without errors
- [ ] Queue throughput: 120 videos/hour
- [ ] Average latency: <60s for 10s @ 1080p

### Week 3
- [ ] Authentication: 0 security issues
- [ ] Billing: $0 failed charges
- [ ] RunPod: Cost <$0.01 per video

### Week 4
- [ ] 100+ users signed up
- [ ] 85%+ success rate in production
- [ ] <1% error rate
- [ ] $500+ MRR from Pro subscriptions

---

## Post-MVP Roadmap

### Month 2: Scale & Quality
- 4K support (tiling/chunking)
- Batch processing (multiple videos)
- Custom SAM 2 fine-tuning for Sora logos
- Video preview before/after comparison

### Month 3: Enterprise
- API access for developers
- Bulk upload (drag folder)
- Whitelabel solution
- SLA guarantees

### Month 4: Expansion
- Support other watermarks (Pika, Runway)
- Mobile app (iOS/Android)
- Plugin ecosystem (Premiere, DaVinci)

---

## Emergency Contacts & Resources

**ComfyUI Issues**: https://github.com/comfyanonymous/ComfyUI/issues
**SAM 2 Docs**: https://github.com/facebookresearch/segment-anything-2
**DiffuEraser Repo**: https://github.com/ShmuelRonen/ComfyUI_DiffuEraser
**sirioberati Workflow**: https://github.com/sirioberati/sora-2-watermark-remover

**GPU Specs**:
- 3070: 8GB VRAM, ~70 videos/hour @ 1080p
- 3090: 24GB VRAM, ~120 videos/hour @ 1080p

**Estimated Costs** (Month 1):
- RunPod serverless: ~$50 (overflow traffic)
- Cloudflare R2: ~$5 (100GB storage)
- Upstash Redis: $0 (free tier)
- Vercel: $0 (hobby plan)
- Clerk: $0 (free tier)
- **Total**: ~$55/month + electricity for 3090

---

**Last Updated**: 2025-10-06
**Status**: Ready for Week 1 execution
