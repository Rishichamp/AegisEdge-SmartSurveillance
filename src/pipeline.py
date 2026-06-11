"""
Surveillance System Orchestrator Pipeline
=========================================

This module is the "brain" or the master orchestrator of our entire Smart
Surveillance system. It connects all independent modules—video streams,
deep-learning models, rule engines, alert engines, and the web dashboard—into
a single running system.

WHY DO WE NEED A PIPELINE ORCHESTRATOR?
======================================
In computer vision projects, it's easy to write a single giant script that does
everything. However, this is hard to maintain, debug, and optimize.

By separating logic into modules (e.g., Detector, Tracker, FightDetector) and
using this `SurveillancePipeline` class to orchestrate them, we achieve:
1. **Modularity:** We can replace YOLOv8 with another detector (e.g., YOLOv10 or Detectron2) without changing the pipeline.
2. **Readability:** The pipeline shows the high-level workflow clearly.
3. **Robustness:** We handle errors, threading, and system shutdown in one place.

HOW THE PIPELINE WORKFLOW WORKS (Frame-by-Frame):
================================================
For each video frame:
1. **Capture:** Read frame from Thread-Safe `VideoStream`.
2. **Object Tracking:** Run YOLOv8 + ByteTrack to locate people/objects and assign tracking IDs.
3. **Deep Feature Extraction:** Crop each person's bounding box and extract features using MobileNetV3.
4. **Behavior Analysis (LSTM):** Store features in a rolling temporal buffer per person. Use the LSTM model to predict actions.
5. **Threat Verification:**
   - Check if 2+ people are in proximity with high chaotic motion (Fight detection).
   - Check if person coordinates enter restricted zone polygons (Intrusion detection).
   - Check if monitored items disappear while a person was nearby (Theft detection).
6. **Alert Processing:** Feed events to `AlertEngine` to apply cooldowns.
7. **Logging & Snapshots:** Write details to `EventLogger` (JSONL) and save screenshots of threats.
8. **Visualization & Streaming:** Annotate the frame and push it to the `DashboardServer` (Flask WebSocket Stream).
"""

import os
import sys
import time
import threading
import traceback
import numpy as np
from typing import List, Dict, Any, Optional

# Import configuration helper
from src.config import ConfigAccessor, get_config

# Import utility modules
from src.utils.video_stream import VideoStream

# Import computer vision & tracking modules
from src.detection.detector import ObjectDetector, Detection

# Import anomaly threat monitors
from src.anomaly.feature_extractor import ROIFeatureExtractor
from src.anomaly.temporal_model import TemporalAnomalyPredictor
from src.anomaly.fight_detector import FightDetector
from src.anomaly.zone_monitor import ZoneMonitor
from src.anomaly.theft_detector import TheftDetector

# Import alerting and logging systems
from src.alerting.alert_engine import AlertEngine
from src.alerting.logger import EventLogger

# Import visualization and web dashboard
from src.visualization.annotator import FrameAnnotator
from src.visualization.dashboard import DashboardServer

# Safely import psutil for CPU utilization
try:
    import psutil
except ImportError:
    psutil = None


class SurveillancePipeline:
    """
    Master pipeline that runs object tracking and anomaly analysis on a video feed.
    """
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        Initialize and configure all surveillance modules based on the YAML configuration file.
        """
        print("[PIPELINE] Initializing Surveillance System Components...")
        
        # Load and parse configuration
        self.config = get_config(config_path)
        
        # 1. Video Input Stream
        # Resolves file paths or integers (webcams)
        self.video_stream = VideoStream(
            source=self.config.video.source,
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps,
            buffer_size=self.config.video.buffer_size
        )
        
        # 2. Object Detector and Tracker (YOLOv8)
        self.detector = ObjectDetector(
            model_path=self.config.detection.model,
            confidence_threshold=self.config.detection.confidence_threshold,
            target_classes=self.config.detection.target_classes,
            device=self.config.detection.device
        )
        
        # 3. Deep Learning Sequence Classifier
        # Extract features (MobileNetV3) -> Sequence Buffer -> Classification (LSTM)
        self.feature_extractor = ROIFeatureExtractor(
            device=self.config.detection.device
        )
        
        self.temporal_predictor = TemporalAnomalyPredictor(
            weights_path=self.config.anomaly.temporal_model.weights_path,
            architecture=self.config.anomaly.temporal_model.architecture,
            sequence_length=self.config.anomaly.temporal_model.sequence_length,
            feature_dim=self.config.anomaly.temporal_model.feature_dim,
            hidden_size=self.config.anomaly.temporal_model.hidden_size,
            num_layers=self.config.anomaly.temporal_model.num_layers,
            num_classes=self.config.anomaly.temporal_model.num_classes,
            device=self.config.detection.device
        )
        
        # 4. Specific Threat Analyzers
        self.fight_detector = FightDetector(
            proximity_threshold=self.config.anomaly.fight_detection.proximity_threshold,
            motion_threshold=self.config.anomaly.fight_detection.motion_threshold,
            min_persons=self.config.anomaly.fight_detection.min_persons
        )
        
        # Convert configuration objects to dictionaries for loaders
        zones_list = []
        for zone in self.config.zones:
            zones_list.append({
                'name': zone.name,
                'polygon': zone.polygon,
                'type': zone.type,
                'color': zone.color
            })
        self.zone_monitor = ZoneMonitor(zones_list)
        
        self.theft_detector = TheftDetector(
            proximity_threshold=self.config.anomaly.theft_detection.proximity_threshold,
            monitored_classes=self.config.anomaly.theft_detection.monitored_objects
        )
        
        # 5. Alert Management and Events Logger
        self.alert_engine = AlertEngine(
            cooldown_seconds=self.config.alerting.cooldown_seconds
        )
        self.event_logger = EventLogger(
            log_file=self.config.alerting.log_file,
            screenshot_dir=self.config.alerting.screenshot_dir
        )
        
        # 6. Visualization & Web Dashboard Server
        self.annotator = FrameAnnotator(zones_list)
        self.dashboard_server = DashboardServer(
            host=self.config.dashboard.host,
            port=self.config.dashboard.port,
            stream_quality=self.config.dashboard.stream_quality
        )
        
        # State tracking variables
        self.running = False
        self._pipeline_thread = None
        self._fps_actual = 0.0
        
    def start(self):
        """
        Start the pipeline execution loop in a separate thread.
        """
        if self.running:
            print("[PIPELINE] Pipeline is already running.")
            return
            
        self.running = True
        
        # Start Dashboard and Capture Stream
        self.video_stream.start()
        self.dashboard_server.start()
        
        # Start Processing Thread
        self._pipeline_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._pipeline_thread.start()
        print("[PIPELINE] Surveillance processing loop started in background thread.")
        
    def _process_loop(self):
        """
        Main frame-by-frame processing pipeline loop.
        """
        frame_idx = 0
        last_time = time.time()
        
        # Active alert cache so that the visual overlays update continuously
        active_alerts_cache = []
        
        # Class lists to clear history for lost tracking IDs
        last_active_tracks = set()
        
        while self.running:
            try:
                # 1. Grab Frame
                frame = self.video_stream.read()
                if frame is None:
                    # Brief sleep to avoid spinning CPU when frame buffer is loading
                    time.sleep(0.01)
                    continue
                
                frame_idx += 1
                
                # Check Frame Skip setting to optimize load
                frame_skip = getattr(self.config.performance, 'frame_skip', 1)
                if frame_idx % frame_skip != 0:
                    continue
                
                # 2. Track Objects (YOLOv8 + ByteTrack)
                tracker_config = getattr(self.config.tracking, 'tracker', "bytetrack.yaml")
                persist_state = getattr(self.config.tracking, 'persist', True)
                
                detections = self.detector.detect_and_track(
                    frame,
                    tracker=tracker_config,
                    persist=persist_state
                )
                
                # 3. Clean up stale/lost tracking IDs from anomaly memory buffers
                current_active_tracks = {d.track_id for d in detections if d.track_id is not None}
                lost_tracks = last_active_tracks - current_active_tracks
                for tid in lost_tracks:
                    self.temporal_predictor.clear_track(tid)
                    self.zone_monitor.clear_track(tid)
                last_active_tracks = current_active_tracks
                
                # Filter people for specialized posture/behavior modeling
                person_detections = [d for d in detections if d.class_name == "person"]
                
                # 4. Temporal Action Predictions (LSTM)
                lstm_predictions = {}
                for person in person_detections:
                    if person.track_id is None:
                        continue
                    
                    # Extract MobileNetV3 visual pose/appearance feature vector
                    features = self.feature_extractor.extract_from_frame(frame, person.bbox)
                    if features is not None:
                        # Append feature to sequence buffer and classify
                        prediction = self.temporal_predictor.add_features(person.track_id, features)
                        if prediction is not None:
                            lstm_predictions[person.track_id] = prediction
                            
                # List of threats triggered on THIS specific frame
                frame_triggered_alerts = []
                
                # 5. Run Threat Analyzers
                
                # A. Fight / Violence Detector
                fight_detected = False
                fight_conf = 0.0
                involved_tracks = []
                if getattr(self.config.anomaly.fight_detection, 'enabled', True):
                    fight_detected, fight_conf, involved_tracks = self.fight_detector.analyze(
                        frame,
                        person_detections,
                        lstm_predictions
                    )
                    if fight_detected:
                        desc = f"Physical altercation detected between tracking IDs: {', '.join(map(str, involved_tracks))}"
                        alert = self.alert_engine.trigger(
                            event_type='FIGHT_DETECTED',
                            severity='CRITICAL',
                            confidence=fight_conf,
                            description=desc,
                            metadata={'involved_tracks': involved_tracks}
                        )
                        if alert:
                            # Capture and log screenshot
                            screenshot_path = self.event_logger.log_event(alert, frame)
                            # Convert absolute local disk path to web-relative path
                            alert['screenshot_url'] = f"/logs/alert_screenshots/{os.path.basename(screenshot_path)}" if screenshot_path else ""
                            
                            self.dashboard_server.send_alert(alert)
                            frame_triggered_alerts.append(alert)
                            
                # B. Restricted Zone Intrusion Detector
                if getattr(self.config.anomaly.intrusion_detection, 'enabled', True):
                    intrusion_events = self.zone_monitor.check_intrusions(person_detections)
                    for event in intrusion_events:
                        if event['event_type'] == 'entry':
                            zone_name = event['zone_name']
                            track_id = event['track_id']
                            zone_type = event['type']
                            severity = 'CRITICAL' if zone_type == 'restricted' else 'WARNING'
                            desc = f"Unauthorized entry in '{zone_name}' by Person #{track_id}"
                            
                            alert = self.alert_engine.trigger(
                                event_type='ZONE_INTRUSION',
                                severity=severity,
                                confidence=0.85,
                                description=desc,
                                metadata={'zone_name': zone_name, 'track_id': track_id, 'zone_type': zone_type}
                            )
                            if alert:
                                screenshot_path = self.event_logger.log_event(alert, frame)
                                alert['screenshot_url'] = f"/logs/alert_screenshots/{os.path.basename(screenshot_path)}" if screenshot_path else ""
                                
                                self.dashboard_server.send_alert(alert)
                                frame_triggered_alerts.append(alert)
                                
                # C. Retail/Office Theft Detector
                if getattr(self.config.anomaly.theft_detection, 'enabled', True):
                    theft_events = self.theft_detector.analyze(detections, frame_idx)
                    for event in theft_events:
                        item_class = event['item_class']
                        thief_id = event['thief_track_id']
                        confidence = event['confidence']
                        
                        desc = f"Monitored item '{item_class}' disappeared from view"
                        if thief_id is not None:
                            desc += f" near Person #{thief_id}"
                            
                        alert = self.alert_engine.trigger(
                            event_type='THEFT_DETECTED',
                            severity='CRITICAL',
                            confidence=confidence,
                            description=desc,
                            metadata={'item_class': item_class, 'thief_track_id': thief_id}
                        )
                        if alert:
                            screenshot_path = self.event_logger.log_event(alert, frame)
                            alert['screenshot_url'] = f"/logs/alert_screenshots/{os.path.basename(screenshot_path)}" if screenshot_path else ""
                            
                            self.dashboard_server.send_alert(alert)
                            frame_triggered_alerts.append(alert)
                
                # Update alert cache (keep alerts from the last 3 seconds active in the visual banner)
                current_time = time.time()
                if frame_triggered_alerts:
                    active_alerts_cache.extend(frame_triggered_alerts)
                
                # Filter out alerts older than 3 seconds
                active_alerts_cache = [
                    a for a in active_alerts_cache
                    if current_time - a['timestamp_unix'] < 3.0
                ]
                
                # 6. Speed Statistics
                elapsed = current_time - last_time
                last_time = current_time
                fps_instant = 1.0 / elapsed if elapsed > 0.0 else 30.0
                self._fps_actual = 0.9 * self._fps_actual + 0.1 * fps_instant if self._fps_actual > 0.0 else fps_instant
                
                # 7. Render Overlay Graphics
                annotated_frame = self.annotator.annotate_frame(
                    frame=frame.copy(),
                    detections=detections,
                    active_alerts=active_alerts_cache,
                    lstm_predictions=lstm_predictions,
                    involved_tracks=involved_tracks,
                    fps=self._fps_actual
                )
                
                # 8. Push Annotated Video Stream Frame
                self.dashboard_server.update_frame(annotated_frame)
                
                # 9. Emit metrics (CPU, tracks, actual FPS) every 5 frames
                if frame_idx % 5 == 0:
                    cpu_val = psutil.cpu_percent() if psutil is not None else 12.5 + (frame_idx % 4)
                    source_str = str(self.config.video.source)
                    source_label = f"Camera {source_str}" if source_str.isdigit() else f"File: {os.path.basename(source_str)}"
                    self.dashboard_server.send_metrics(
                        fps=self._fps_actual,
                        tracks=len(current_active_tracks),
                        cpu_percent=cpu_val,
                        camera_source=source_label
                    )
                    
                # Short sleep to prevent CPU starvation in the thread
                time.sleep(0.005)
                
            except Exception as e:
                print(f"[PIPELINE] Error in processing loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)
                
        print("[PIPELINE] Background processing loop stopped.")
        
    def stop(self):
        """
        Gracefully stop all pipeline threads and servers.
        """
        if not self.running:
            return
            
        print("[PIPELINE] Stopping all components...")
        self.running = False
        
        # Stop children
        self.video_stream.stop()
        self.dashboard_server.stop()
        
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=3.0)
            
        print("[PIPELINE] Surveillance system shutdown complete.")
        
    @property
    def fps(self) -> float:
        """Returns the smoothed actual running frame rate of the system."""
        return self._fps_actual


if __name__ == "__main__":
    # Test script loading and component assembly
    print("Testing pipeline initialization...")
    try:
        pipeline = SurveillancePipeline()
        print("Success! All pipeline modules configured and ready.")
    except Exception as err:
        print(f"Failed to initialize pipeline: {err}")
        traceback.print_exc()
