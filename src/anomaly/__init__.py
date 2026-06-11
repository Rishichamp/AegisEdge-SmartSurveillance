"""
Anomaly Detection Module
========================
AI-powered behavior analysis for detecting fights, theft, and intrusion.

Components:
    - feature_extractor.py:  CNN feature extraction from person ROIs
    - temporal_model.py:     LSTM-based temporal activity classifier
    - fight_detector.py:     Violence/fight detection logic
    - zone_monitor.py:       Zone-based intrusion detection
    - theft_detector.py:     Theft behavior detection
"""
from .feature_extractor import ROIFeatureExtractor
from .temporal_model import TemporalAnomalyPredictor
from .fight_detector import FightDetector
from .zone_monitor import ZoneMonitor
from .theft_detector import TheftDetector
