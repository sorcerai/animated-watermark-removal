# Modal Deployment Prep

This folder contains the Modal app harness for running the Sora watermark
removal pipeline on on-demand GPUs.

## Prerequisites

1. Export your tuned ComfyUI workflow as
   `workflows/sora-removal-production.json`. Run `python3 scripts/validate_workflow.py`
   to confirm the graph includes the required nodes.
2. Install dependencies locally (needed for `modal deploy`):

   ```bash
   pip install -r requirements.txt
   ```

3. Provision a Modal secret named `sora-watermark-remover` that stores any
   Cloudflare R2 (or S3-compatible) credentials you plan to use:

   ```bash
   modal secret create sora-watermark-remover \
     --env R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
     --env R2_ACCESS_KEY_ID=... \
     --env R2_SECRET_ACCESS_KEY=... \
     --env R2_BUCKET=sora-watermark-output
   ```

   If you prefer to skip uploads, omit the variables—the function will simply
   return the local summary.

4. Ensure ComfyUI is reachable from the Modal worker. Set
   `COMFY_HOST` accordingly (for example via Modal secret or `modal env`).

## Deploy & Test

```bash
modal deploy deploy/modal/app.py
modal run deploy/modal/app.py --test-video-url "https://example.com/clip.mp4"
```

To call the GPU function from another service:

```python
import modal

app = modal.App.lookup("sora-watermark-remover")
process_clip = app.function("process_clip")

result = process_clip.remote(
    video_url="https://storage/clip.mp4",
    sam_point=(960, 540),
    output_key="user123/clip.mp4",
)
```

The response includes:

```json
{
  "summary": { ... job metadata ... },
  "upload_url": "https://.../signed"
}
```

## Notes

- The GPU tier defaults to `A10G`. Adjust inside `deploy/modal/app.py` if you
  need more VRAM (e.g., `gpu="A100"`).
- The function downloads the source clip into a temp directory, runs the same
  orchestration pipeline used locally, then optionally uploads the final MP4 to
  Cloudflare R2.
- Logs are written to `/tmp/.../logs`, included in the summary JSON. Since the
  temp directory is cleaned up after execution, rely on the upload or log data
  rather than the local file path.
