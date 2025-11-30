# System Architecture

**Last Updated**: 2025-10-06

---

## Overview

Sora Watermark Remover is a hybrid SaaS platform that combines:
- **sirioberati's proven ComfyUI workflow** (SAM 2 + DiffuEraser) for watermark removal
- **Custom automation layer** for async processing, queue management, and scaling
- **Hybrid GPU strategy** (own 3090 + serverless overflow) for cost-effective scaling

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Layer (Frontend)                        │
│  ┌─────────────────────┐      ┌─────────────────────────────────┐  │
│  │  Next.js Upload UI  │      │   SAM Point Selector (Canvas)   │  │
│  │  - Drag & Drop      │      │   - First frame thumbnail       │  │
│  │  - Progress Bar     │      │   - Click to mark logo center   │  │
│  └─────────────────────┘      └─────────────────────────────────┘  │
│               │                              │                       │
│               └──────────────┬───────────────┘                       │
│                              │ HTTPS                                 │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                   API Layer (Node.js/Fastify)                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  POST /api/upload          │  GET /api/status/:jobId           │ │
│  │  - Validate format         │  - Query job queue                │ │
│  │  - Check quota (Clerk)     │  - Return progress %              │ │
│  │  - Save temp file          │                                   │ │
│  │  - Enqueue job             │  GET /api/download/:jobId         │ │
│  │                            │  - Generate signed R2 URL         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                    ┌─────────┼─────────┐                            │
│                    │    Redis Queue    │                            │
│                    │    (BullMQ)       │                            │
│                    │                   │                            │
│                    │  ┌─────────────┐ │                            │
│                    │  │ Job 1: 🔄  │ │  Max 3 concurrent (3090)    │
│                    │  │ Job 2: ⏳  │ │  Overflow → RunPod          │
│                    │  │ Job 3: ⏳  │ │                              │
│                    │  └─────────────┘ │                            │
│                    └─────────┬─────────┘                            │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌────────────────┐   ┌────────────────┐   ┌─────────────────┐
│  3090 Worker   │   │  3090 Worker   │   │  RunPod Worker  │
│  (Local)       │   │  (Local)       │   │  (Serverless)   │
│                │   │                │   │                 │
│  ┌──────────┐  │   │  ┌──────────┐  │   │  ┌───────────┐ │
│  │ ComfyUI  │  │   │  │ ComfyUI  │  │   │  │ ComfyUI   │ │
│  │ + SAM2   │  │   │  │ + SAM2   │  │   │  │ + SAM2    │ │
│  │ +DiffuE  │  │   │  │ +DiffuE  │  │   │  │ +DiffuE   │ │
│  └──────────┘  │   │  └──────────┘  │   │  └───────────┘ │
│       │        │   │       │        │   │        │        │
│  1. Extract    │   │  1. Extract    │   │  1. Extract     │
│  2. Segment    │   │  2. Segment    │   │  2. Segment     │
│  3. Inpaint    │   │  3. Inpaint    │   │  3. Inpaint     │
│  4. Stitch     │   │  4. Stitch     │   │  4. Stitch      │
└────────┬───────┘   └────────┬───────┘   └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  Cloudflare R2 Storage     │
                │                            │
                │  /user-id/video-id/        │
                │    ├─ input.mp4            │
                │    ├─ output.mp4           │
                │    └─ thumbnail.jpg        │
                │                            │
                │  Lifecycle:                │
                │  - 24h → archive           │
                │  - 7d → delete             │
                └────────────────────────────┘
```

---

## Core Components

### 1. Frontend (Next.js)

**Responsibilities**:
- Video upload with drag & drop
- SAM 2 point selection (interactive canvas)
- Real-time progress tracking via polling
- Download link generation

**Tech Stack**:
- **Framework**: Next.js 14 (App Router)
- **Auth**: @clerk/nextjs
- **Upload**: react-dropzone
- **State**: zustand
- **Hosting**: Vercel (edge functions for API routes)

**Key Files**:
- `app/upload/page.tsx` - Main upload interface
- `app/status/[jobId]/page.tsx` - Progress tracking page
- `components/SAMPointSelector.tsx` - Interactive point selection
- `lib/api-client.ts` - API wrapper

---

### 2. API Layer (Node.js)

**Responsibilities**:
- Handle uploads and validation
- Enforce quotas and authentication
- Manage job queue
- Generate signed download URLs

**Tech Stack**:
- **Framework**: Fastify (faster than Express)
- **Queue**: BullMQ + Redis
- **Auth**: @clerk/clerk-sdk-node
- **Storage**: @aws-sdk/client-s3 (R2 compatible)
- **Hosting**: Railway/Render

**Endpoints**:

```typescript
POST   /api/upload
       Body: { file: File, samPoint: [x, y] }
       Returns: { jobId: string, status: 'queued' }

GET    /api/status/:jobId
       Returns: { status: 'queued'|'processing'|'completed'|'failed',
                  progress: number,
                  estimatedTime: number }

GET    /api/download/:jobId
       Returns: { downloadUrl: string (signed, expires 1h) }

POST   /api/checkout
       Body: { plan: 'pro'|'enterprise' }
       Returns: { checkoutUrl: string (Stripe) }
```

**Key Files**:
- `src/api/routes/upload.ts`
- `src/api/routes/status.ts`
- `src/api/routes/download.ts`
- `src/api/middleware/auth.ts`
- `src/api/middleware/quota.ts`

---

### 3. Processing Queue (BullMQ)

**Responsibilities**:
- Async job processing
- Concurrency control (max 3 for 3090)
- Retry logic (max 2 attempts)
- Progress tracking

**Queue Configuration**:

```typescript
const videoQueue = new Queue('video-processing', {
    connection: { host: 'localhost', port: 6379 },
    defaultJobOptions: {
        attempts: 2,
        backoff: {
            type: 'exponential',
            delay: 5000
        },
        removeOnComplete: {
            age: 86400  // 24 hours
        }
    }
});

const worker = new Worker('video-processing', processVideo, {
    connection: { host: 'localhost', port: 6379 },
    concurrency: 3  // 3090 can handle 3 concurrent @ 1080p
});
```

**Job Lifecycle**:

```
Queued → Processing → [Extract → Segment → Inpaint → Stitch] → Upload → Completed
   │                                                                        │
   └──────────────── Failed (retry) ──────────────────────────────────────┘
```

**Key Files**:
- `src/api/queue/video-queue.ts`
- `src/api/queue/workers/video-worker.ts`
- `src/api/queue/jobs/process-video.ts`

---

### 4. ComfyUI Client (Python → Node.js bridge)

**Responsibilities**:
- Connect to ComfyUI API (WebSocket + HTTP)
- Upload videos and queue workflows
- Monitor progress via WebSocket
- Download processed frames

**Architecture**:

```
Node.js Worker
     │
     ├─→ spawn Python subprocess
     │       │
     │       └─→ comfyui_client.py
     │               │
     │               ├─ HTTP: POST /upload/image
     │               ├─ HTTP: POST /prompt (queue workflow)
     │               ├─ WebSocket: /ws (progress updates)
     │               └─ HTTP: GET /view (download result)
     │
     └─← receive JSON result
```

**Python Client** (`src/comfyui/client.py`):

```python
class ComfyUIClient:
    def __init__(self, host="http://localhost:8188"):
        self.host = host
        self.ws_url = host.replace('http', 'ws') + '/ws'

    def process_video(
        self,
        video_path: str,
        workflow_path: str,
        sam_point: tuple[int, int],
        callback: Optional[Callable[[int], None]] = None
    ) -> dict:
        """
        Process video through ComfyUI workflow

        Steps:
        1. Upload video via /upload/image
        2. Load workflow JSON
        3. Inject SAM point into workflow
        4. Submit to /prompt endpoint
        5. Monitor WebSocket for progress
        6. Download result frames from /view

        Returns:
            {
                "prompt_id": str,
                "status": "success" | "failed",
                "output_frames": str,  # Path to frames directory
                "processing_time": float
            }
        """
        # Implementation details in Week 2
```

**Node.js Wrapper** (`src/comfyui/client.ts`):

```typescript
import { spawn } from 'child_process';

export class ComfyUIClient {
    async processVideo(
        videoPath: string,
        samPoint: [number, number],
        onProgress?: (progress: number) => void
    ): Promise<{ outputFrames: string }> {
        return new Promise((resolve, reject) => {
            const python = spawn('python', [
                'src/comfyui/client.py',
                videoPath,
                JSON.stringify(samPoint)
            ]);

            python.stdout.on('data', (data) => {
                const event = JSON.parse(data.toString());
                if (event.type === 'progress') {
                    onProgress?.(event.progress);
                } else if (event.type === 'complete') {
                    resolve({ outputFrames: event.outputFrames });
                }
            });

            python.stderr.on('data', (data) => {
                console.error(data.toString());
            });

            python.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`ComfyUI failed with code ${code}`));
                }
            });
        });
    }
}
```

---

### 5. Audio Preservation Pipeline

**Critical for SaaS**: Audio must be bit-perfect copy of original, synced within 1 frame (<33ms @ 30fps)

**Pipeline**:

```python
# src/core/audio_preservation.py

class AudioPreservationPipeline:
    def extract_frames(self, video_path: str) -> Path:
        """
        Extract frames using FFmpeg
        - Quality: -qscale:v 2 (high quality PNG)
        - Format: %06d.png (000001.png, 000002.png, ...)
        - Metadata: Save FPS for later stitching
        """

    def stitch_with_audio(
        self,
        frames_dir: str,
        original_video: str,
        output_video: str
    ) -> Path:
        """
        Stitch frames back with original audio

        CRITICAL FLAGS:
        - `-map 0:v` = video from processed frames
        - `-map 1:a` = audio from original video
        - `-c:a copy` = COPY audio, no re-encoding

        This ensures bit-perfect audio preservation
        """
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", str(frames_dir / "%06d.png"),  # Processed frames
            "-i", str(original_video),  # Original for audio
            "-map", "0:v",  # Video from frames
            "-map", "1:a",  # Audio from original
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "copy",  # <<< CRITICAL
            "-movflags", "+faststart",
            str(output_video)
        ]

    def validate_audio_sync(
        self,
        original: str,
        processed: str
    ) -> dict:
        """
        Validate audio sync

        Checks:
        - Duration difference < 33ms (1 frame @ 30fps)
        - Audio stream present
        - No corruption

        Returns:
            {
                "synced": bool,
                "duration_diff_ms": float,
                "original_has_audio": bool,
                "processed_has_audio": bool
            }
        """
```

**Why This Matters**:
- Users expect perfect audio sync (any desync is immediately noticeable)
- Re-encoding audio (`-c:a aac`) introduces artifacts and sync issues
- Using `-c:a copy` preserves original bitstream exactly

---

### 6. Storage Layer (Cloudflare R2)

**Why R2 over S3**:
- **Cost**: $0.015/GB storage (10x cheaper than S3)
- **Bandwidth**: $0 egress (S3 charges $0.09/GB)
- **API**: S3-compatible (drop-in replacement)

**Storage Structure**:

```
sora-watermark-videos/
  ├─ user-123/
  │   ├─ video-abc/
  │   │   ├─ input.mp4
  │   │   ├─ output.mp4
  │   │   └─ thumbnail.jpg
  │   └─ video-def/
  │       ├─ input.mp4
  │       └─ output.mp4
  └─ user-456/
      └─ video-ghi/
          ├─ input.mp4
          └─ output.mp4
```

**Lifecycle Policies**:

```typescript
// Automatically applied by R2
const lifecycleRules = [
    {
        // Archive to cheaper storage after 24h
        id: 'archive-old-videos',
        status: 'Enabled',
        transitions: [{
            days: 1,
            storageClass: 'INFREQUENT_ACCESS'
        }]
    },
    {
        // Delete after 7 days
        id: 'delete-old-videos',
        status: 'Enabled',
        expiration: {
            days: 7
        }
    }
];
```

**Signed URLs**:

```typescript
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { GetObjectCommand } from '@aws-sdk/client-s3';

async function getDownloadUrl(key: string): Promise<string> {
    const command = new GetObjectCommand({
        Bucket: 'sora-watermark-videos',
        Key: key
    });

    // URL expires in 1 hour
    return await getSignedUrl(r2Client, command, {
        expiresIn: 3600
    });
}
```

---

### 7. GPU Scaling Strategy

**Hybrid Approach**: Own hardware (predictable cost) + serverless (elastic scaling)

#### Scenario 1: Low Traffic (0-50 videos/day)

```
All jobs → 3090 (local)
Cost: $0 (electricity only ~$5/month)
Latency: <60s for 10s video @ 1080p
```

#### Scenario 2: Medium Traffic (50-500 videos/day)

```
First 3 concurrent → 3090 (local)
Overflow → RunPod Serverless
Cost: ~$50/month RunPod + $5 electricity
Latency: <90s for 10s video (serverless cold start)
```

#### Scenario 3: High Traffic (500+ videos/day)

```
Queue routing logic:
- If 3090 available → use local
- If 3090 busy → RunPod
- If RunPod cold → spin up instance (15-30s)

Cost: ~$200/month RunPod + $5 electricity
Latency: <60s average (mostly warm instances)
```

**Cost Comparison**:

```
Own 3090:
- Electricity: ~$5/month (24/7 idle)
- Processing: $0 per video
- Throughput: 120 videos/hour

RunPod Serverless (RTX 4090):
- Cost: $0.39/hour (billed per second)
- Processing: ~$0.006 per 10s video
- Throughput: 150 videos/hour per instance
- Cold start: 15-30s

Break-even: ~850 videos/month
```

**Decision Logic** (`src/api/queue/workers/gpu-router.ts`):

```typescript
async function selectGPU(job: VideoJob): Promise<'local' | 'runpod'> {
    const queueLength = await videoQueue.count();
    const local3090Available = queueLength < 3;

    // Priority 1: Use local if available (free)
    if (local3090Available) {
        return 'local';
    }

    // Priority 2: Check if user is Pro/Enterprise (better experience)
    const user = await getUser(job.userId);
    if (user.tier !== 'free') {
        return 'runpod';  // Faster for paid users
    }

    // Priority 3: Queue for free users (cost optimization)
    return 'local';  // Wait for 3090
}
```

---

## Data Flow: End-to-End

```
1. USER UPLOADS VIDEO
   ├─ Frontend: Drag & drop MP4 → upload to /api/upload
   ├─ API: Save to temp storage, enqueue job
   └─ Response: { jobId: "abc123" }

2. USER SELECTS SAM POINT
   ├─ Frontend: Show first frame, user clicks logo center
   ├─ API: Update job with samPoint: [960, 540]
   └─ Queue: Job ready for processing

3. QUEUE WORKER PICKS UP JOB
   ├─ Check GPU availability (local vs RunPod)
   ├─ Download video from temp storage
   └─ Start processing

4. EXTRACT FRAMES
   ├─ FFmpeg: video.mp4 → frames/ (PNG sequence)
   ├─ Save FPS metadata for later
   └─ Duration: ~5s for 10s video

5. COMFYUI PROCESSING
   ├─ Upload frames to ComfyUI
   ├─ Load workflow: sora-removal-production.json
   ├─ Inject SAM point into workflow nodes
   ├─ Queue workflow via /prompt
   ├─ Monitor WebSocket for progress (0-100%)
   │   ├─ 0-20%: SAM 2 segmentation (generate masks)
   │   ├─ 20-80%: DiffuEraser inpainting (remove watermark)
   │   └─ 80-100%: Temporal smoothing
   ├─ Download processed frames from /view
   └─ Duration: ~40s for 10s video @ 1080p

6. STITCH WITH AUDIO
   ├─ FFmpeg: frames/ + original.mp4 → output.mp4
   ├─ Flags: -c:a copy (bit-perfect audio)
   ├─ Validate sync: <33ms tolerance
   └─ Duration: ~5s

7. UPLOAD TO R2
   ├─ Upload output.mp4 to R2
   ├─ Generate thumbnail (first frame)
   ├─ Delete temp files
   └─ Duration: ~3s

8. NOTIFY USER
   ├─ Update job status: "completed"
   ├─ Frontend polls /api/status → sees completion
   ├─ Frontend requests /api/download → gets signed URL
   └─ User downloads video

TOTAL TIME: ~60s for 10s video @ 1080p
```

---

## Security Considerations

### Authentication
- **Clerk** for user management (OAuth, magic links)
- **JWT tokens** for API requests
- **Session cookies** for web requests

### File Upload Safety
- **Format validation**: Only MP4/MOV/AVI accepted
- **Size limits**: 500MB max (prevents abuse)
- **Virus scanning**: ClamAV on upload (optional, post-MVP)
- **Quota enforcement**: Free tier limited to 3/month

### API Security
- **Rate limiting**: 10 uploads/hour per user
- **CORS**: Restrict to frontend domain only
- **Content-Type validation**: Multipart form-data only
- **Signed URLs**: Download links expire after 1 hour

### Data Privacy
- **Encryption at rest**: R2 default encryption
- **Encryption in transit**: HTTPS only
- **Data retention**: Auto-delete after 7 days
- **No logging**: Videos not stored in logs

---

## Monitoring & Observability

### Error Tracking
- **Sentry**: Capture exceptions in API and workers
- **LogRocket**: Session replay for frontend debugging
- **Custom alerts**: Notify on >5% error rate

### Performance Metrics
- **Queue length**: Alert if >10 jobs waiting
- **Processing time**: P50, P95, P99 latencies
- **GPU utilization**: Track VRAM usage via nvidia-smi
- **Success rate**: % of jobs completed successfully

### Business Metrics
- **Signups**: Daily new users
- **Conversions**: Free → Pro upgrades
- **Revenue**: MRR from Stripe
- **Churn**: Cancelled subscriptions

### Dashboards
- **Grafana**: Real-time metrics
- **Plausible**: Privacy-friendly analytics
- **Stripe Dashboard**: Revenue tracking

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Production                               │
│                                                                  │
│  Frontend (Vercel)                                              │
│    ├─ CDN: Cloudflare                                           │
│    ├─ Domain: app.sorawatermarkremover.com                      │
│    └─ SSL: Auto (Vercel)                                        │
│                                                                  │
│  API (Railway)                                                  │
│    ├─ Region: US-West (close to 3090)                           │
│    ├─ Scaling: 2-5 instances                                    │
│    └─ Health checks: /health every 30s                          │
│                                                                  │
│  Redis (Upstash)                                                │
│    ├─ Plan: Free tier (10K commands/day)                        │
│    └─ Persistence: AOF enabled                                  │
│                                                                  │
│  3090 Worker (Dedicated Windows Box)                            │
│    ├─ ComfyUI: http://localhost:8188                            │
│    ├─ Tunneling: ngrok/Cloudflare Tunnel                        │
│    ├─ Monitoring: UptimeRobot                                   │
│    └─ Backup: Daily model/workflow backups                      │
│                                                                  │
│  RunPod Serverless                                              │
│    ├─ Template: Custom (ComfyUI + models)                       │
│    ├─ Region: US-East, US-West                                  │
│    ├─ Auto-scaling: 0-10 instances                              │
│    └─ Cost: $0.39/hour per instance                             │
│                                                                  │
│  Storage (Cloudflare R2)                                        │
│    ├─ Bucket: sora-watermark-videos                             │
│    ├─ Region: Auto (distributed)                                │
│    └─ Lifecycle: 7-day auto-delete                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Decisions & Rationale

### Why ComfyUI over custom pipeline?
- ✅ **Proven workflow**: sirioberati's approach tested by community
- ✅ **Visual debugging**: Easy to tune parameters in GUI
- ✅ **Model flexibility**: Easy to swap SAM 2 Large ↔ Base
- ❌ **API overhead**: Adds ~500ms latency vs native Python
- ❌ **Windows-first**: Most CI/CD assumes Linux

**Decision**: Use ComfyUI, optimize later if needed

### Why Node.js API over Python (FastAPI)?
- ✅ **Ecosystem**: Better integrations (Clerk, Stripe, Vercel)
- ✅ **Async I/O**: Better for queue coordination
- ✅ **Frontend synergy**: Same language for Next.js
- ❌ **ML libraries**: Worse than Python ecosystem

**Decision**: Node.js for API, Python for ComfyUI client

### Why BullMQ over RabbitMQ/Kafka?
- ✅ **Simple**: Redis-backed, no JVM required
- ✅ **Features**: Built-in retry, rate limiting, priority
- ✅ **Monitoring**: Bull Board for web UI
- ❌ **Scaling**: Not for millions of jobs/sec (overkill for MVP)

**Decision**: BullMQ, migrate to Kafka if needed at scale

### Why Cloudflare R2 over S3?
- ✅ **Cost**: $0 egress (S3 charges $0.09/GB)
- ✅ **Global**: Cloudflare's edge network
- ❌ **Maturity**: Newer service, fewer features

**Decision**: R2 for bandwidth savings, fallback to S3 if issues

### Why hybrid GPU over full serverless?
- ✅ **Cost**: Own 3090 = $0 per video (electricity only)
- ✅ **Latency**: Local = no cold start (15-30s)
- ✅ **Control**: Full access for debugging
- ❌ **Availability**: If 3090 crashes, 100% serverless
- ❌ **Scaling**: Limited to 120 videos/hour

**Decision**: Hybrid for cost optimization, serverless safety net

---

**Last Updated**: 2025-10-06
**Review**: Before Week 2 (API implementation)
