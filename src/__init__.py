"""
Smart Surveillance System
========================

A real-time AI-powered surveillance system with object detection,
multi-object tracking, and anomaly detection (fights, theft, intrusion).

Package Structure:
    src/
    ├── detection/       # YOLOv8 object detection & tracking
    ├── anomaly/         # Anomaly detection modules
    ├── models/          # Neural network architectures (LSTM, 3D CNN)
    ├── alerting/        # Alert generation & logging
    ├── visualization/   # Frame annotation & web dashboard
    ├── edge/            # Edge deployment utilities
    ├── utils/           # Shared utilities
    ├── config.py        # Configuration loader
    └── pipeline.py      # Main pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "Smart Surveillance System Contributors"
