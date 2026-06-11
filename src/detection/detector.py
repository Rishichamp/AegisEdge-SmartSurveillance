"""
Object Detection Wrapper (YOLOv8)
==================================

This module wraps the Ultralytics YOLOv8 library. It provides classes
to perform object detection and tracking on video frames.

WHY YOLOv8?
===========
YOLO (You Only Look Once) is the industry standard for real-time object detection.
Version 8 offers state-of-the-art accuracy and speed, native tracking support
(ByteTrack/BoT-SORT), and supports exporting to various formats (ONNX, TensorRT).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """
    Data class representing a single object detection.
    """
    bbox: np.ndarray        # Coordinates: [x1, y1, x2, y2]
    confidence: float       # Confidence score (0.0 to 1.0)
    class_id: int           # COCO class ID
    class_name: str         # COCO class name (e.g. "person")
    track_id: Optional[int] = None  # Tracker ID (if tracking is enabled)


class ObjectDetector:
    """
    YOLOv8 Object Detector and Tracker wrapper.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to PyTorch (.pt), ONNX (.onnx), or TensorRT (.engine) weights.
            confidence_threshold: Minimum confidence score to accept detection.
            target_classes: List of COCO class IDs to filter. Defaults to None (detect all).
                            Common IDs: 0=person, 24=backpack, 26=handbag, 28=suitcase,
                                        39=bottle, 63=laptop, 67=cell phone
            device: Device to run inference on ("cpu", "cuda", "0", etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.device = device
        
        # Load the YOLO model
        print(f"[DETECTOR] Loading model weights from: {model_path} on {device}")
        self.model = YOLO(model_path)
        
        # Class names lookup mapping (class_id -> name)
        self.class_names = self.model.names
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform standard object detection on a single frame.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # bbox coordinates: [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            
            detections.append(Detection(
                bbox=xyxy,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))
            
        return detections
        
    def detect_and_track(
        self,
        frame: np.ndarray,
        tracker: str = "bytetrack.yaml",
        persist: bool = True
    ) -> List[Detection]:
        """
        Perform object detection AND assign persistent tracking IDs across frames.
        Uses YOLOv8's built-in tracker wrapper.
        
        Args:
            frame: Video frame (BGR NumPy array)
            tracker: Tracker config filename ("bytetrack.yaml" or "botsort.yaml")
            persist: Keep tracking state from previous frames (essential for video)
        """
        # YOLOv8 track command handles tracking.
        # It takes care of mapping boxes across frames using Kalman filters/ByteTrack.
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            classes=self.target_classes,
            device=self.device,
            tracker=tracker,
            persist=persist,
            verbose=False
        )
        
        detections = []
        if len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            
            # Obtain track ID if tracking succeeded
            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
            
            detections.append(Detection(
                bbox=xyxy,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
                track_id=track_id
            ))
            
        return detections
