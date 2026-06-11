"""
Unit Tests for Track History Manager
====================================

This module tests the historical tracking path builder, track clearing,
and velocity/speed calculation physics.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.detection.tracker import TrackHistoryManager
from src.detection.detector import Detection


class TestTrackHistoryManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = TrackHistoryManager(max_history=10)
        
    def test_update_and_trail(self):
        """Verify tracking history queues update correctly."""
        # Frame 1: Person #1 at (100, 100) -> bbox center
        bbox1 = np.array([90, 90, 110, 110])
        det1 = Detection(bbox=bbox1, confidence=0.85, class_id=0, class_name="person", track_id=1)
        
        self.manager.update([det1], frame_index=1)
        
        trail = self.manager.get_trail(1)
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0], (100.0, 100.0))
        
        # Frame 2: Person #1 moves to (110, 110)
        bbox2 = np.array([100, 100, 120, 120])
        det2 = Detection(bbox=bbox2, confidence=0.85, class_id=0, class_name="person", track_id=1)
        
        self.manager.update([det2], frame_index=2)
        
        trail = self.manager.get_trail(1)
        self.assertEqual(len(trail), 2)
        self.assertEqual(trail[1], (110.0, 110.0))
        
    def test_history_cap(self):
        """Verify the history queue respects max_history limit."""
        for i in range(15):
            bbox = np.array([i, i, i+20, i+20])
            det = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person", track_id=1)
            self.manager.update([det], frame_index=i)
            
        trail = self.manager.get_trail(1)
        self.assertEqual(len(trail), 10)  # max_history limit
        
    def test_velocity_and_speed(self):
        """Verify movement velocity vectors and speed magnitudes."""
        # Simulate moving diagonally at 10 pixels per frame
        for i in range(5):
            pos = i * 10.0
            bbox = np.array([pos-10, pos-10, pos+10, pos+10])
            det = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person", track_id=1)
            self.manager.update([det], frame_index=i)
            
        # Velocity over window = 5: dx = (40 - 0) = 40, dy = (40 - 0) = 40. Frames elapsed = 4
        # dx/frame = 10, dy/frame = 10
        dx, dy = self.manager.calculate_velocity(1, window=5)
        self.assertAlmostEqual(dx, 10.0)
        self.assertAlmostEqual(dy, 10.0)
        
        # Speed = sqrt(10^2 + 10^2) = sqrt(200) = ~14.14
        speed = self.manager.calculate_speed(1, window=5)
        self.assertAlmostEqual(speed, np.sqrt(200.0))
        
    def test_stale_track_cleanup(self):
        """Tracks not seen in 60+ frames should be cleaned up from memory."""
        bbox = np.array([0, 0, 20, 20])
        det = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        # Update at frame 1
        self.manager.update([det], frame_index=1)
        self.assertIn(1, self.manager.tracks)
        
        # Update with empty list at frame 62 (elapsed 61 frames)
        self.manager.update([], frame_index=62)
        
        # Memory should be cleaned
        self.assertNotIn(1, self.manager.tracks)
        self.assertNotIn(1, self.manager.last_seen)


if __name__ == '__main__':
    unittest.main()
