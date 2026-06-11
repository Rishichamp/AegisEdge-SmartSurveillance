"""
Detection Module
================
Object detection (YOLOv8) and multi-object tracking (ByteTrack).

Components:
    - detector.py:  YOLOv8 wrapper for real-time object detection
    - tracker.py:   Multi-object tracker for persistent ID assignment
"""
from .detector import ObjectDetector, Detection
from .tracker import TrackHistoryManager
