"""Modal deployment scaffold for the Sora watermark remover pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYS_PATH_ROOT = Path("/app/src")

if str(SYS_PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_PATH_ROOT))


image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/app/src")
    .add_local_dir(PROJECT_ROOT / "workflows", remote_path="/app/workflows")
)

models_path = PROJECT_ROOT / "models"
if models_path.exists():
    image = image.add_local_dir(models_path, remote_path="/app/models")


app = modal.App("sora-watermark-remover")


def _download_video(url: str, target_path: Path) -> Path:
    import requests

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with target_path.open("wb") as fout:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fout.write(chunk)

    return target_path


def _upload_to_r2(file_path: Path, key: str) -> str:
    import boto3

    endpoint = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")

    if not all([endpoint, access_key, secret_key, bucket]):
        raise RuntimeError("R2 credentials are not fully configured")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    with file_path.open("rb") as stream:
        client.upload_fileobj(stream, bucket, key)

    signed_url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )

    return signed_url


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    secrets=[modal.Secret.from_name("sora-watermark-remover")],
)
def process_clip(
    video_url: str,
    sam_point: Optional[Tuple[int, int]] = None,
    output_key: Optional[str] = None,
) -> dict:
    """Entry point executed on Modal GPU infrastructure."""

    from pipeline.job import PipelineConfig, SoraWatermarkJob
    from comfyui import ComfyUIClient

    comfy_host = os.getenv("COMFY_HOST", "http://127.0.0.1:8188")
    workflow_path = Path("/app/workflows/sora-removal-production.json")

    if not workflow_path.exists():
        raise FileNotFoundError(
            "Workflow missing on remote. Upload sora-removal-production.json first."
        )

    with tempfile.TemporaryDirectory(prefix="sora-modal-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.mp4"

        _download_video(video_url, input_path)

        config = PipelineConfig(
            workflow_path=workflow_path,
            working_root=tmp_path / "working",
            output_root=tmp_path / "output",
            logs_root=tmp_path / "logs",
            overwrite=False,
            backup_original=False,
            keep_temp=True,
        )

        job = SoraWatermarkJob(
            input_video=input_path,
            config=config,
            comfy_client=ComfyUIClient(host=comfy_host),
            sam_point=sam_point,
        )

        summary = job.run()

        output_path = Path(summary["output_video"])
        upload_url = None

        if output_key:
            upload_url = _upload_to_r2(output_path, output_key)

        return {
            "summary": summary,
            "upload_url": upload_url,
        }


@app.local_entrypoint()
def main(test_video_url: str):
    """Simple smoke test local entrypoint."""

    result = process_clip.local(test_video_url)
    print(result)
