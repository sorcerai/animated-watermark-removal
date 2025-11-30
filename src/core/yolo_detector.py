#!/usr/bin/env python3
"""
YOLO-World Open-Vocabulary Detector for Animated Watermark Detection

Uses YOLO-World's open-vocabulary capability to detect:
- "SORA" text watermark
- @username variations (animated watermark)
- General text/watermark patterns

Returns bounding boxes for SAM3 segmentation.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class YOLOWorldDetector:
    """
    YOLO-World wrapper for open-vocabulary watermark detection.

    YOLO-World can detect objects by text description without
    pre-training on specific classes - perfect for detecting
    "SORA" text and @username variations.
    """

    # Default detection classes for animated watermarks
    DEFAULT_CLASSES = [
        "SORA",
        "text",
        "watermark",
        "logo",
        "username",
        "brand text",
    ]

    def __init__(
        self,
        model_path: Union[str, Path],
        classes: Optional[List[str]] = None,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize YOLO-World detector.

        Args:
            model_path: Path to YOLO-World weights (.pt file)
            classes: List of text classes to detect (open vocabulary)
            device: Device to run on ("cuda" or "cpu")
            conf_threshold: Minimum confidence threshold
            iou_threshold: NMS IoU threshold
        """
        self.model_path = Path(model_path)
        self.classes = classes or self.DEFAULT_CLASSES
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO-World model from ultralytics."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO-World from {self.model_path}")
            self.model = YOLO(str(self.model_path))

            # Set detection classes (open vocabulary)
            self.model.set_classes(self.classes)

            # Move to device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    logger.info("YOLO-World loaded on CUDA")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.device = "cpu"
            else:
                logger.info("YOLO-World loaded on CPU")

        except ImportError:
            raise ImportError(
                "ultralytics is required. Install: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO-World model: {e}")

    def set_classes(self, classes: List[str]):
        """
        Update detection classes at runtime.

        Useful for switching between detection targets.

        Args:
            classes: New list of text classes to detect
        """
        self.classes = classes
        if self.model is not None:
            self.model.set_classes(classes)
            logger.info(f"Updated detection classes: {classes}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        return_raw: bool = False
    ) -> List[Dict]:
        """
        Detect watermarks/text in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, C)
            conf_threshold: Override default confidence threshold
            return_raw: If True, also return raw ultralytics results

        Returns:
            List of detections, each with:
            - bbox: [x1, y1, x2, y2] pixel coordinates
            - conf: confidence score (0-1)
            - class_id: detected class index
            - class_name: detected class name
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        conf = conf_threshold or self.conf_threshold

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                box = boxes[i]

                # Extract bbox coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Get class name
                class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"

                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                }

                detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x["conf"], reverse=True)

        if return_raw:
            return detections, results
        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Detect watermarks in multiple frames.

        Args:
            frames: List of BGR images
            conf_threshold: Override default confidence threshold

        Returns:
            List of detection lists (one per frame)
        """
        conf = conf_threshold or self.conf_threshold

        # Batch inference
        results = self.model.predict(
            source=frames,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            stream=True  # Memory efficient for large batches
        )

        all_detections = []

        for result in results:
            frame_detections = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i]

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"

                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    frame_detections.append(detection)

            frame_detections.sort(key=lambda x: x["conf"], reverse=True)
            all_detections.append(frame_detections)

        return all_detections

    def get_best_detection(
        self,
        detections: List[Dict],
        prefer_classes: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Get the best detection from a list.

        Args:
            detections: List of detections from detect()
            prefer_classes: Prioritize these class names

        Returns:
            Best detection dict, or None if empty
        """
        if not detections:
            return None

        if prefer_classes:
            # First try to find preferred class with high confidence
            for det in detections:
                if det["class_name"] in prefer_classes:
                    return det

        # Otherwise return highest confidence
        return detections[0]

    def merge_overlapping_detections(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Merge overlapping detections into single bounding boxes.

        Useful when SORA and @username are detected as separate boxes
        but should be treated as one watermark region.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for merging

        Returns:
            Merged list of detections
        """
        if len(detections) <= 1:
            return detections

        def iou(box1, box2):
            """Calculate IoU between two boxes."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter

            return inter / union if union > 0 else 0

        # Group overlapping detections
        merged = []
        used = [False] * len(detections)

        for i, det1 in enumerate(detections):
            if used[i]:
                continue

            # Start new group with this detection
            group = [det1]
            used[i] = True

            for j, det2 in enumerate(detections[i+1:], i+1):
                if used[j]:
                    continue

                if iou(det1["bbox"], det2["bbox"]) > iou_threshold:
                    group.append(det2)
                    used[j] = True

            # Merge group into single bbox (union of all boxes)
            if len(group) > 1:
                x1 = min(d["bbox"][0] for d in group)
                y1 = min(d["bbox"][1] for d in group)
                x2 = max(d["bbox"][2] for d in group)
                y2 = max(d["bbox"][3] for d in group)

                merged_det = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": max(d["conf"] for d in group),
                    "class_id": group[0]["class_id"],
                    "class_name": "merged_watermark"
                }
                merged.append(merged_det)
            else:
                merged.append(det1)

        return merged

    def __repr__(self):
        return (
            f"YOLOWorldDetector("
            f"model={self.model_path.name}, "
            f"classes={self.classes}, "
            f"device={self.device})"
        )
