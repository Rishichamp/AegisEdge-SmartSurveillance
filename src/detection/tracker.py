"""
Object Tracking History Manager
===============================

This module provides track history management, helping us follow objects over
time, calculate their speed/velocity vectors, and maintain trajectory paths.

WHY A TRACK MANAGER?
====================
YOLOv8's built-in tracker assigns IDs to boxes on a frame-by-frame basis,
but it does not store past coordinate history. To detect temporal anomalies
(like running, crawling, or sudden movements in a fight), we need to store
the history of coordinates for each unique tracker ID.
"""

import collections
from typing import Dict, List, Tuple
import numpy as np


class TrackHistoryManager:
    """
    Manages historical positions of tracked objects.
    Useful for visualizing trails, calculating speeds, and detecting motion patterns.
    """
    
    def __init__(self, max_history: int = 30):
        """
        Args:
            max_history: Maximum number of frames to store coordinates for each object.
                         At 30 FPS, 30 frames = 1 second of history.
        """
        self.max_history = max_history
        # Dict mapping track_id -> deque of (x, y) coordinates
        self.tracks = {}
        # Dict mapping track_id -> timestamp or frame index when last seen
        self.last_seen = {}
        
    def update(self, detections: List[any], frame_index: int):
        """
        Update the history with the current frame's detections.
        
        Args:
            detections: List of Detection objects with active track_ids.
            frame_index: Index of the current frame (for cleaning old tracks).
        """
        active_ids = set()
        
        for det in detections:
            if det.track_id is None:
                continue
                
            track_id = det.track_id
            active_ids.add(track_id)
            
            # Find bottom center or center of bounding box
            x1, y1, x2, y2 = det.bbox[:4]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = collections.deque(maxlen=self.max_history)
                
            self.tracks[track_id].append(center)
            self.last_seen[track_id] = frame_index
            
        # Clean up old tracks that haven't been seen in over 60 frames (2 seconds)
        inactive_ids = []
        for track_id, last_idx in self.last_seen.items():
            if frame_index - last_idx > 60:
                inactive_ids.append(track_id)
                
        for track_id in inactive_ids:
            if track_id in self.tracks:
                del self.tracks[track_id]
            del self.last_seen[track_id]
            
    def get_trail(self, track_id: int) -> List[Tuple[float, float]]:
        """Get the historical trail of a track ID."""
        if track_id in self.tracks:
            return list(self.tracks[track_id])
        return []
        
    def calculate_velocity(self, track_id: int, window: int = 5) -> Tuple[float, float]:
        """
        Calculate the velocity vector (dx, dy) of an object over a window of frames.
        
        Args:
            track_id: The ID of the object.
            window: Number of recent frames to use for calculation.
            
        Returns:
            (dx, dy) representing pixels per frame.
        """
        trail = self.get_trail(track_id)
        if len(trail) < 2:
            return (0.0, 0.0)
            
        # Use a sub-window
        pts = trail[-min(len(trail), window):]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        
        frames_elapsed = len(pts) - 1
        return (dx / frames_elapsed, dy / frames_elapsed)
        
    def calculate_speed(self, track_id: int, window: int = 5) -> float:
        """
        Calculate speed of a track ID (magnitude of velocity).
        """
        dx, dy = self.calculate_velocity(track_id, window)
        return float(np.sqrt(dx**2 + dy**2))
