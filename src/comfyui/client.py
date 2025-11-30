#!/usr/bin/env python3
"""
ComfyUI API Client

Handles communication with ComfyUI server for video processing:
- Upload videos and frames
- Queue workflows with custom parameters
- Monitor progress via WebSocket
- Download processed results

Usage:
    client = ComfyUIClient()
    result = client.process_video(
        video_path="input.mp4",
        workflow_path="workflows/sora-removal-production.json",
        sam_point=(960, 540),
        callback=lambda p: print(f"Progress: {p}%")
    )
"""

import requests
import json
import websocket
import uuid
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
import threading


class ComfyUIClient:
    def __init__(self, host: str = "http://localhost:8188"):
        self.host = host
        self.ws_url = host.replace('http', 'ws') + '/ws'
        self.client_id = str(uuid.uuid4())

    def upload_file(self, file_path: str, subfolder: str = "") -> dict:
        """
        Upload file to ComfyUI server

        Args:
            file_path: Path to file
            subfolder: Optional subfolder in ComfyUI/input

        Returns:
            {"name": str, "subfolder": str, "type": str}
        """
        file_path = Path(file_path)

        with open(file_path, 'rb') as f:
            files = {'image': (file_path.name, f, 'application/octet-stream')}
            data = {'subfolder': subfolder, 'type': 'input'}

            response = requests.post(
                f"{self.host}/upload/image",
                files=files,
                data=data
            )
            response.raise_for_status()

        return response.json()

    def queue_workflow(
        self,
        workflow: dict,
        sam_point: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Queue workflow for processing

        Args:
            workflow: Workflow JSON (from sirioberati)
            sam_point: SAM 2 point prompt (x, y) for logo center

        Returns:
            prompt_id: Unique ID for this job
        """
        # Inject SAM point into workflow if provided
        if sam_point:
            workflow = self._inject_sam_point(workflow, sam_point)

        prompt = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        response = requests.post(
            f"{self.host}/prompt",
            json=prompt
        )
        response.raise_for_status()

        return response.json()['prompt_id']

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for prompt"""
        response = requests.get(f"{self.host}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    def get_queue(self) -> dict:
        """Get current queue status"""
        response = requests.get(f"{self.host}/queue")
        response.raise_for_status()
        return response.json()

    def monitor_progress(
        self,
        prompt_id: str,
        callback: Optional[Callable[[int], None]] = None
    ) -> dict:
        """
        Monitor workflow execution via WebSocket

        Args:
            prompt_id: Job ID to monitor
            callback: Optional progress callback (0-100)

        Returns:
            Final execution result
        """
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}")

        progress_value = 0

        try:
            while True:
                message = json.loads(ws.recv())
                msg_type = message.get('type')

                if msg_type == 'execution_start':
                    if callback:
                        callback(progress_value)

                elif msg_type == 'executing':
                    data = message.get('data', {})
                    prompt = data.get('prompt_id')

                    if prompt == prompt_id:
                        # Estimate progress based on node completion
                        # (rough heuristic, can be improved)
                        if callback:
                            # Simple progress: each node = ~10%
                            # Actual progress depends on workflow structure
                            progress_value = min(95, progress_value + 5)
                            callback(progress_value)

                elif msg_type == 'execution_cached':
                    # Node was cached, skip
                    pass

                elif msg_type == 'execution_error':
                    error = message.get('data', {})
                    raise RuntimeError(f"ComfyUI error: {error}")

                elif msg_type == 'executed':
                    data = message.get('data', {})
                    prompt = data.get('prompt_id')

                    if prompt == prompt_id:
                        if callback:
                            callback(100)

                        # Get final output
                        history = self.get_history(prompt_id)
                        return history[prompt_id]

        finally:
            ws.close()

    def download_output(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output"
    ) -> bytes:
        """
        Download processed file from ComfyUI

        Args:
            filename: Output filename
            subfolder: Subfolder in ComfyUI/output
            folder_type: Usually "output" or "temp"

        Returns:
            File content as bytes
        """
        params = {
            'filename': filename,
            'subfolder': subfolder,
            'type': folder_type
        }

        response = requests.get(
            f"{self.host}/view",
            params=params
        )
        response.raise_for_status()

        return response.content

    def process_video(
        self,
        video_path: str,
        workflow_path: str,
        sam_point: Tuple[int, int],
        callback: Optional[Callable[[int], None]] = None
    ) -> dict:
        """
        Process video through ComfyUI workflow (end-to-end)

        Args:
            video_path: Path to input video
            workflow_path: Path to workflow JSON
            sam_point: SAM 2 point prompt (x, y)
            callback: Optional progress callback

        Returns:
            {
                "prompt_id": str,
                "status": "success" | "failed",
                "output_frames": str,  # Path to frames directory
                "processing_time": float
            }
        """
        start_time = time.time()

        # 1. Upload video
        if callback:
            callback(5)

        upload_result = self.upload_file(video_path)
        print(f"✓ Uploaded {video_path}")

        # 2. Load workflow
        workflow = json.loads(Path(workflow_path).read_text())
        print(f"✓ Loaded workflow: {workflow_path}")

        # 3. Update workflow with uploaded video
        workflow = self._update_workflow_input(
            workflow,
            upload_result['name'],
            upload_result.get('subfolder', '')
        )

        # 4. Queue workflow
        if callback:
            callback(10)

        prompt_id = self.queue_workflow(workflow, sam_point)
        print(f"✓ Queued workflow: {prompt_id}")

        # 5. Monitor progress
        result = self.monitor_progress(prompt_id, callback)

        # 6. Get output info
        outputs = result.get('outputs', {})

        # Find output node (usually SaveVideo or SaveImage)
        output_frames = None
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                # Found output frames
                output_frames = node_output['images']
                break

        processing_time = time.time() - start_time

        return {
            "prompt_id": prompt_id,
            "status": "success" if output_frames else "failed",
            "output_frames": output_frames,
            "processing_time": processing_time
        }

    def _inject_sam_point(self, workflow: dict, sam_point: Tuple[int, int]) -> dict:
        """
        Inject SAM 2 point prompt into workflow

        Finds SAM2 nodes and updates point coordinates
        """
        workflow = json.loads(json.dumps(workflow))  # deep copy to avoid side-effects

        coordinates_json = json.dumps([
            {"x": int(sam_point[0]), "y": int(sam_point[1])}
        ])

        nodes = workflow.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                node_type = node.get("type") or node.get("class_type")

                if node_type in {"Sam2VideoSegmentationAddPoints", "Sam2Segmentation"}:
                    widgets = node.get("widgets_values", [])

                    if node_type == "Sam2VideoSegmentationAddPoints":
                        if widgets:
                            widgets[0] = coordinates_json
                        else:
                            widgets = [coordinates_json, 0, 0]
                    elif node_type == "Sam2Segmentation":
                        # Sam2Segmentation widgets: [auto_mask, bboxes, positive, keep]
                        while len(widgets) < 4:
                            widgets.append("")
                        widgets[2] = coordinates_json

                    node["widgets_values"] = widgets

        return workflow

    def _update_workflow_input(
        self,
        workflow: dict,
        filename: str,
        subfolder: str
    ) -> dict:
        """
        Update workflow to use uploaded video/image

        Finds LoadVideo or LoadImage nodes
        """
        workflow = workflow.copy()

        for node_id, node in workflow.items():
            class_type = node.get('class_type')

            if class_type in ['LoadVideo', 'LoadImage', 'LoadVideoPath']:
                inputs = node.get('inputs', {})
                inputs['video'] = filename
                if subfolder:
                    inputs['subfolder'] = subfolder
                node['inputs'] = inputs

        return workflow


def main():
    """Test client"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python client.py <video_path> <sam_point_json>")
        print('Example: python client.py input.mp4 "[960, 540]"')
        sys.exit(1)

    video_path = sys.argv[1]
    sam_point = tuple(json.loads(sys.argv[2]))

    client = ComfyUIClient()

    def progress_callback(p):
        print(f"Progress: {p}%", flush=True)

    result = client.process_video(
        video_path=video_path,
        workflow_path="workflows/templates/sora-removal-production.json",
        sam_point=sam_point,
        callback=progress_callback
    )

    # Output JSON for Node.js to parse
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
