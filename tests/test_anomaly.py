"""
Unit Tests for Anomaly Detectors
===============================

This module tests the proximity, optical flow, and disappearance analytics
in the FightDetector and TheftDetector classes.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.anomaly.fight_detector import FightDetector
from src.anomaly.theft_detector import TheftDetector
from src.detection.detector import Detection


class TestAnomalyDetectors(unittest.TestCase):
    
    # ----------------------------
    # 1. Fight Detector Tests
    # ----------------------------
    def test_fight_detector_proximity(self):
        """Verify fight detector rule-based person proximity calculations."""
        # 150 pixel threshold
        detector = FightDetector(proximity_threshold=150.0, min_persons=2)
        
        # Test Case A: No people (min_persons not met)
        close_pairs = detector.check_proximity([])
        self.assertEqual(len(close_pairs), 0)
        
        # Test Case B: 2 people far apart (centers at (50, 50) and (300, 50) -> distance = 250)
        p1 = np.array([40, 40, 60, 60])
        p2 = np.array([290, 40, 310, 60])
        close_pairs = detector.check_proximity([p1, p2])
        self.assertEqual(len(close_pairs), 0)
        
        # Test Case C: 2 people close (centers at (50, 50) and (100, 50) -> distance = 50)
        p3 = np.array([90, 40, 110, 60])
        close_pairs = detector.check_proximity([p1, p3])
        self.assertEqual(len(close_pairs), 1)
        self.assertEqual(close_pairs[0][0], 0)  # Person index 0
        self.assertEqual(close_pairs[0][1], 1)  # Person index 1
        self.assertAlmostEqual(close_pairs[0][2], 50.0) # Distance
        
    def test_fight_analysis_rules(self):
        """Verify fight detector fuses proximity, motion energy, and model predictions."""
        # Setup detector
        detector = FightDetector(proximity_threshold=100.0, motion_threshold=10.0, lstm_weight=0.5, rules_weight=0.5)
        
        # Scenario: People in proximity, but no motion (static frames)
        # Mock calculate_motion_energy to return 0
        detector.calculate_motion_energy = lambda frame: 0.0
        
        p1 = Detection(bbox=np.array([40, 40, 60, 60]), confidence=0.9, class_id=0, class_name="person", track_id=1)
        p2 = Detection(bbox=np.array([80, 40, 100, 60]), confidence=0.9, class_id=0, class_name="person", track_id=2)
        
        # Frame array (placeholder)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        is_fight, score, tracks = detector.analyze(frame, [p1, p2], lstm_predictions={})
        # Should not flag fight due to zero motion
        self.assertFalse(is_fight)
        
        # Scenario B: People close + high motion
        detector.calculate_motion_energy = lambda frame: 12.0 # Exceeds threshold of 10.0
        is_fight_2, score_2, tracks_2 = detector.analyze(frame, [p1, p2], lstm_predictions={})
        
        self.assertTrue(is_fight_2)
        self.assertIn(1, tracks_2)
        self.assertIn(2, tracks_2)
        
    # ----------------------------
    # 2. Theft Detector Tests
    # ----------------------------
    def test_theft_detector_analysis(self):
        """Verify theft detector monitors disappearing items near people."""
        # 120 pixels proximity, 2 frames threshold for quick testing
        detector = TheftDetector(proximity_threshold=120.0, disappearance_frames=2)
        
        # Frame 1: Person (track #1) is close to a laptop. Laptop is detected.
        # Person center: (50, 50). Laptop center: (100, 50). Dist = 50.
        person = Detection(bbox=np.array([40, 20, 60, 80]), confidence=0.9, class_id=0, class_name="person", track_id=1)
        laptop = Detection(bbox=np.array([90, 40, 110, 60]), confidence=0.9, class_id=63, class_name="laptop")
        
        alerts = detector.analyze([person, laptop], frame_idx=1)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(len(detector.stationary_items), 1)
        
        # Verify stationary item was saved with last seen near person 1
        item_key = list(detector.stationary_items.keys())[0]
        self.assertEqual(detector.stationary_items[item_key]['last_seen_near_person'], 1)
        
        # Frame 2: Laptop disappears (not in detection list).
        alerts_2 = detector.analyze([person], frame_idx=2)
        self.assertEqual(len(alerts_2), 0)  # Missing for 1 frame (threshold is 2)
        
        # Frame 3: Laptop still missing (missing for 2 frames). Alert should trigger.
        alerts_3 = detector.analyze([person], frame_idx=3)
        self.assertEqual(len(alerts_3), 1)
        self.assertEqual(alerts_3[0]['event_type'], 'theft')
        self.assertEqual(alerts_3[0]['item_class'], 'laptop')
        self.assertEqual(alerts_3[0]['thief_track_id'], 1)
        
        # Verify it was removed from active monitoring to prevent duplicate alerts
        self.assertEqual(len(detector.stationary_items), 0)


if __name__ == '__main__':
    unittest.main()
