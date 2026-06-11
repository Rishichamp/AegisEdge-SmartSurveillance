"""
Event Logger
============

This module logs events to a structured JSONL file and saves corresponding
alert screenshots on disk.

WHY STRUCTURED JSON LOGS?
=========================
A JSONL (JSON Lines) file is a file where each line is a valid JSON object.
This format is excellent for high-volume logs:
1. Append-only: We can log a new event by appending a line without rewriting the file.
2. Easy to parse line-by-line (scalable).
3. Easily consumed by modern log analysis platforms (ELK stack, Datadog).
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, Optional


class EventLogger:
    """
    Logs structured events and manages screenshots folder.
    """
    
    def __init__(self, log_file: str = "logs/events.jsonl", screenshot_dir: str = "logs/alert_screenshots"):
        self.log_file = log_file
        self.screenshot_dir = screenshot_dir
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
        
    def log_event(self, alert_event: Dict[str, Any], frame: Optional[np.ndarray] = None) -> str:
        """
        Log an event to the JSONL file and optionally capture/save a screenshot.
        
        Args:
            alert_event: The dictionary returned by the AlertEngine.
            frame: The current BGR image frame.
            
        Returns:
            The path to the saved screenshot (if saved), otherwise an empty string.
        """
        screenshot_path = ""
        
        # Save screenshot if frame is provided
        if frame is not None:
            # Construct file name based on event type and timestamp
            sanitized_type = alert_event['event_type'].lower().replace(' ', '_')
            filename = f"{alert_event['timestamp'].replace(':', '-')}_{sanitized_type}.jpg"
            screenshot_path = os.path.join(self.screenshot_dir, filename)
            
            # Save file using OpenCV
            cv2.imwrite(screenshot_path, frame)
            # Add path to the event record
            alert_event['metadata']['screenshot_path'] = screenshot_path
            
        # Append to JSONL log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert_event) + '\n')
        except Exception as e:
            print(f"[LOGGER] Error writing to log file: {e}")
            
        return screenshot_path
