"""
Unit Tests for Master Surveillance Pipeline
===========================================

This module tests the initialization and setup states of the Master Pipeline.
It mocks the deep-learning model load steps (YOLOv8) to enable fast, offline,
and sandboxed execution of unit tests.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestSurveillancePipeline(unittest.TestCase):
    
    @patch('src.detection.detector.YOLO')
    def test_pipeline_initialization(self, mock_yolo):
        """Verify the pipeline orchestrator loads config and initializes child components."""
        # 1. Setup Mock YOLO to prevent downloading PT weights or loading model graph
        mock_instance = MagicMock()
        mock_instance.names = {
            0: "person", 24: "backpack", 26: "handbag", 
            28: "suitcase", 63: "laptop", 67: "cell phone"
        }
        mock_yolo.return_value = mock_instance
        
        # 2. Import pipeline and instantiate
        from src.pipeline import SurveillancePipeline
        pipeline = SurveillancePipeline(config_path="config/default_config.yaml")
        
        # 3. Verify configurations loaded successfully
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.config.video.width, 640)
        self.assertEqual(pipeline.config.video.height, 480)
        
        # 4. Verify sub-modules were initialized
        self.assertIsNotNone(pipeline.video_stream)
        self.assertIsNotNone(pipeline.detector)
        self.assertIsNotNone(pipeline.feature_extractor)
        self.assertIsNotNone(pipeline.temporal_predictor)
        self.assertIsNotNone(pipeline.fight_detector)
        self.assertIsNotNone(pipeline.zone_monitor)
        self.assertIsNotNone(pipeline.theft_detector)
        self.assertIsNotNone(pipeline.alert_engine)
        self.assertIsNotNone(pipeline.event_logger)
        self.assertIsNotNone(pipeline.annotator)
        self.assertIsNotNone(pipeline.dashboard_server)
        
        # 5. Verify initial states
        self.assertFalse(pipeline.running)
        self.assertEqual(pipeline.fps, 0.0)


if __name__ == '__main__':
    unittest.main()
