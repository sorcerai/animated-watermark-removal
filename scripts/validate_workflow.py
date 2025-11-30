#!/usr/bin/env python3
"""Workflow validator for sora-removal-production.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

REQUIRED_NODE_TYPES = {
    "load_video": ("Video loader", {"LoadVideo", "VHS_LoadVideo", "LoadVideoPath"}),
    "sam2": ("SAM2 segmentation", {"SAM2", "Sam2Segmentation", "Sam2VideoSegmentation", "Sam2VideoSegmentationAddPoints", "Sam2AutoSegmentation"}),
    "diffueraser": ("DiffuEraser inpainting", {"DiffuEraserSampler", "DiffuEraserLoader"}),
    "save": ("Save processed frames", {"SaveImage", "SaveVideo"}),
}


def load_workflow(path: Path) -> Dict:
    try:
        data = json.loads(path.read_text())
        return data
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Workflow JSON is invalid: {exc}") from exc


def _iter_nodes(workflow: Dict) -> Iterable[Dict]:
    if isinstance(workflow, dict):
        if "nodes" in workflow and isinstance(workflow["nodes"], list):
            yield from workflow["nodes"]
        else:
            # Assume mapping of id -> node objects
            yield from workflow.values()
    elif isinstance(workflow, list):  # pragma: no cover - defensive
        yield from workflow
    else:  # pragma: no cover - defensive
        raise ValueError("Unsupported workflow JSON structure")


def assert_required_nodes(workflow: Dict) -> List[str]:
    missing: List[str] = []
    nodes = list(_iter_nodes(workflow))

    for key, (description, acceptable) in REQUIRED_NODE_TYPES.items():
        if not any((node.get("type") or node.get("class_type")) in acceptable for node in nodes):
            missing.append(description)

    return missing


def list_output_nodes(workflow: Dict) -> Iterable[str]:
    for node in _iter_nodes(workflow):
        class_type = node.get("type") or node.get("class_type", "")
        if "Save" in class_type:
            node_id = node.get("id") or node.get("_id") or "?"
            yield f"{node_id}: {class_type}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ComfyUI workflow for the Sora pipeline."
    )
    parser.add_argument(
        "workflow",
        type=Path,
        default=Path("workflows/sora-removal-production.json"),
        help="Path to the workflow JSON exported from ComfyUI.",
    )

    args = parser.parse_args()
    path = args.workflow.expanduser().resolve()

    if not path.exists():
        raise SystemExit(f"Workflow not found: {path}")

    workflow = load_workflow(path)
    missing = assert_required_nodes(workflow)

    if missing:
        print("✗ Missing required nodes:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)

    outputs = list(list_output_nodes(workflow))
    print("✓ Workflow contains required nodes.")
    if outputs:
        print("Output nodes detected:")
        for entry in outputs:
            print(f"  - {entry}")
    else:
        print("⚠ No Save* nodes detected. Ensure frames are exported for stitching.")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
