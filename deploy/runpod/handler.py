"""
RunPod Serverless Handler

Handles video processing requests on RunPod serverless infrastructure
Automatically scales from 0 to N instances based on queue depth

Environment Variables:
- R2_ACCESS_KEY_ID: Cloudflare R2 access key
- R2_SECRET_ACCESS_KEY: Cloudflare R2 secret key
- R2_ENDPOINT: Cloudflare R2 endpoint
- R2_BUCKET: Cloudflare R2 bucket name
"""

import runpod
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from client import ComfyUIClient
from audio_preservation import AudioPreservationPipeline


def download_from_url(url: str) -> str:
    """Download video from URL to temp file"""
    import requests

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(temp_file.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return temp_file.name


def upload_to_r2(file_path: str, key: str) -> str:
    """Upload processed video to Cloudflare R2"""
    import boto3
    import os

    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('R2_ENDPOINT'),
        aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
    )

    bucket = os.getenv('R2_BUCKET')

    with open(file_path, 'rb') as f:
        s3.upload_fileobj(f, bucket, key)

    # Generate signed URL (1 hour expiry)
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600
    )

    return url


def process_video(job):
    """
    Main handler function for RunPod

    Input:
        {
            "video_url": str,  # URL to download video from
            "sam_point": [x, y],  # SAM 2 point prompt
            "user_id": str,  # User ID for R2 path
            "video_id": str  # Video ID for R2 path
        }

    Output:
        {
            "output_url": str,  # Signed R2 URL
            "processing_time": float,
            "status": "success" | "failed"
        }
    """
    try:
        input_data = job['input']

        video_url = input_data['video_url']
        sam_point = tuple(input_data['sam_point'])
        user_id = input_data['user_id']
        video_id = input_data['video_id']

        print(f"Processing video {video_id} for user {user_id}")

        # 1. Download video
        print("Downloading video...")
        video_path = download_from_url(video_url)

        # 2. Extract frames (with audio preservation)
        print("Extracting frames...")
        audio_pipeline = AudioPreservationPipeline()
        frames_dir = audio_pipeline.extract_frames(video_path)

        # 3. Process through ComfyUI
        print("Processing with ComfyUI...")
        client = ComfyUIClient()

        result = client.process_video(
            video_path=video_path,
            workflow_path='/app/ComfyUI/workflows/sora-removal-production.json',
            sam_point=sam_point,
            callback=lambda p: print(f"ComfyUI progress: {p}%")
        )

        if result['status'] != 'success':
            raise RuntimeError("ComfyUI processing failed")

        # 4. Stitch frames with audio
        print("Stitching with audio...")
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        audio_pipeline.stitch_with_audio(
            frames_dir=result['output_frames'],
            original_video=video_path,
            output_video=output_path
        )

        # 5. Validate audio sync
        print("Validating audio sync...")
        validation = audio_pipeline.validate_audio_sync(
            video_path,
            output_path
        )

        if not validation['synced']:
            raise RuntimeError(
                f"Audio desync detected: {validation['duration_diff_ms']}ms"
            )

        # 6. Upload to R2
        print("Uploading to R2...")
        r2_key = f"{user_id}/{video_id}/output.mp4"
        output_url = upload_to_r2(output_path, r2_key)

        # 7. Cleanup temp files
        print("Cleaning up...")
        audio_pipeline.cleanup(frames_dir)
        Path(video_path).unlink()
        Path(output_path).unlink()

        print(f"✓ Processing complete: {result['processing_time']}s")

        return {
            "output_url": output_url,
            "processing_time": result['processing_time'],
            "status": "success"
        }

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "status": "failed",
            "error": str(e)
        }


# Start RunPod serverless handler
runpod.serverless.start({"handler": process_video})
