"""
Unit Tests for Restricted Zone Monitor
======================================

This module tests the coordinate geometry (point-in-polygon) and transition states
(entry, inside, exit) of the Restricted Zone Monitor.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.anomaly.zone_monitor import ZoneMonitor
from src.detection.detector import Detection


class TestZoneMonitor(unittest.TestCase):
    
    def setUp(self):
        # Configure a simple 100x100 square restricted zone at coordinates (100, 100) -> (200, 200)
        self.zones_config = [{
            'name': 'Secure Room',
            'polygon': [[100, 100], [200, 100], [200, 200], [100, 200]],
            'type': 'restricted',
            'color': [0, 0, 255]
        }]
        self.monitor = ZoneMonitor(self.zones_config)
        
    def test_zone_initialization(self):
        """Verify zones configuration parses correctly."""
        self.assertEqual(len(self.monitor.zones), 1)
        zone = self.monitor.zones[0]
        self.assertEqual(zone['name'], 'Secure Room')
        self.assertEqual(zone['type'], 'restricted')
        # Check polygon is structured as tuples of floats
        self.assertEqual(zone['polygon'], [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)])
        
    def test_outside_zone(self):
        """Detections outside the zone should trigger no entry/inside alerts."""
        # Bottom center is (50, 50) which is outside (100, 100) -> (200, 200)
        bbox = np.array([40, 20, 60, 50]) 
        det = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        events = self.monitor.check_intrusions([det])
        self.assertEqual(len(events), 0)
        
    def test_entry_and_presence_transitions(self):
        """Entering the zone should trigger 'entry', and staying should trigger 'inside'."""
        # 1. Entry frame: bottom center is (150, 150) -> inside the zone
        bbox_in = np.array([130, 100, 170, 150]) 
        det = Detection(bbox=bbox_in, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        events_1 = self.monitor.check_intrusions([det])
        self.assertEqual(len(events_1), 1)
        self.assertEqual(events_1[0]['event_type'], 'entry')
        self.assertEqual(events_1[0]['zone_name'], 'Secure Room')
        self.assertEqual(events_1[0]['track_id'], 1)
        
        # 2. Sequential frame: person is still inside the zone
        events_2 = self.monitor.check_intrusions([det])
        self.assertEqual(len(events_2), 1)
        self.assertEqual(events_2[0]['event_type'], 'inside')
        
    def test_exit_transition(self):
        """Moving out of the zone should trigger an 'exit' event."""
        bbox_in = np.array([130, 100, 170, 150])  # inside
        det_in = Detection(bbox=bbox_in, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        # Trigger entry first
        self.monitor.check_intrusions([det_in])
        
        # Move outside
        bbox_out = np.array([300, 300, 340, 350])  # outside
        det_out = Detection(bbox=bbox_out, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        events = self.monitor.check_intrusions([det_out])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'exit')
        
    def test_clear_track(self):
        """Clearing a track ID should purge its persistent history state."""
        bbox_in = np.array([130, 100, 170, 150])
        det = Detection(bbox=bbox_in, confidence=0.9, class_id=0, class_name="person", track_id=1)
        
        # Enter zone
        self.monitor.check_intrusions([det])
        self.assertIn(1, self.monitor.inside_tracker)
        
        # Clear tracker state
        self.monitor.clear_track(1)
        self.assertNotIn(1, self.monitor.inside_tracker)


if __name__ == '__main__':
    unittest.main()
