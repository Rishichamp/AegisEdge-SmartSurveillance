"""
Frame Annotator
===============

This module overlays bounding boxes, zone outlines, active alerts,
and system status statistics onto the video frames.

WHY SEPARATE ANNOTATION?
=========================
Separating visual display rendering from core analytics logic keeps the codebase clean.
The main processing pipeline runs mathematics on coordinates; the annotator only runs
drawing operations.

VISUAL BEST PRACTICES (Aesthetics):
==================================
- Use transparency (alpha blending) for zone overlays so the camera view isn't blocked.
- Avoid flat colors; use a professional color palette.
  * Person: Bright Blue or Green (Normal)
  * Restricted Zone: Translucent Red/Orange
  * Active Alert: Solid red top banner with text shadows
- Clean typography for labels and track IDs.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from src.detection.detector import Detection


class FrameAnnotator:
    """
    Renders visual elements (bounding boxes, zones, alerts, system metrics) on images.
    """
    
    def __init__(self, zones_config: List[Dict[str, Any]]):
        self.zones = zones_config
        
        # Color Palette (BGR format)
        self.colors = {
            'normal_person': (255, 191, 0),    # Deep Sky Blue
            'item': (0, 165, 255),             # Orange
            'alert': (0, 0, 255),              # Red
            'warning': (0, 128, 255),          # Yellow/Orange
            'text_white': (255, 255, 255),
            'banner_bg': (20, 20, 20)          # Very dark grey
        }
        
    def draw_zones(self, frame: np.ndarray, active_zone_alerts: List[str]) -> np.ndarray:
        """
        Draw semi-transparent polygonal zones on the frame.
        """
        overlay = frame.copy()
        
        for zone in self.zones:
            poly = np.array(zone['polygon'], dtype=np.int32).reshape((-1, 1, 2))
            
            # Check if this zone currently has a triggered intrusion
            name = zone.get('name', '')
            is_alerting = name in active_zone_alerts
            
            color = self.colors['alert'] if is_alerting else tuple(zone.get('color', [0, 255, 0]))
            
            # Draw semi-transparent filled polygon
            cv2.fillPoly(overlay, [poly], color)
            
            # Draw solid outline
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
            
            # Draw Zone Label near the first vertex
            x, y = zone['polygon'][0]
            cv2.putText(
                frame,
                f"Zone: {name}",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['text_white'],
                1,
                cv2.LINE_AA
            )
            
        # Blend overlay (alpha transparency = 0.25)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
        
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        involved_tracks: List[int],
        lstm_predictions: Dict[int, Dict[str, float]]
    ):
        """
        Draw bounding boxes and labels for each detection.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            track_id = det.track_id
            
            # Determine color
            if track_id is not None and track_id in involved_tracks:
                color = self.colors['alert']
            elif det.class_name == "person":
                color = self.colors['normal_person']
            else:
                color = self.colors['item']
                
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Construct Label
            label = f"{det.class_name}"
            if track_id is not None:
                label += f" #{track_id}"
                
            # Append LSTM prediction if available for this person
            if track_id is not None and track_id in lstm_predictions:
                preds = lstm_predictions[track_id]
                # Find argmax class
                max_class = max(preds, key=preds.get)
                max_val = preds[max_class]
                if max_class != "normal" and max_val > 0.4:
                    label += f" ({max_class}:{max_val:.1f})"
                    
            # Draw text background banner
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_w, label_h = label_size
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw Label
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.colors['text_white'],
                1,
                cv2.LINE_AA
            )
            
    def draw_alert_banner(self, frame: np.ndarray, active_alerts: List[Dict[str, Any]]):
        """
        Draw an alert notification banner at the top of the frame.
        """
        if not active_alerts:
            return
            
        h, w = frame.shape[:2]
        banner_h = 40
        
        # Sort active alerts by severity (CRITICAL first)
        critical_alerts = [a for a in active_alerts if a['severity'] == 'CRITICAL']
        warning_alerts = [a for a in active_alerts if a['severity'] == 'WARNING']
        
        if critical_alerts:
            color = self.colors['alert']
            alert_text = f"CRITICAL: {critical_alerts[0]['description']}"
        elif warning_alerts:
            color = self.colors['warning']
            alert_text = f"WARNING: {warning_alerts[0]['description']}"
        else:
            return
            
        # Draw banner background
        cv2.rectangle(frame, (0, 0), (w, banner_h), color, -1)
        
        # Draw alert text
        cv2.putText(
            frame,
            alert_text,
            (20, banner_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors['text_white'],
            2,
            cv2.LINE_AA
        )
        
    def draw_status_overlay(self, frame: np.ndarray, fps: float, active_tracks: int):
        """
        Draw metadata (FPS, target count) in the top-right corner.
        """
        h, w = frame.shape[:2]
        stats_w, stats_h = 160, 50
        
        # Dark transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - stats_w, 0), (w, stats_h), self.colors['banner_bg'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - stats_w + 15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.colors['text_white'],
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Active tracks: {active_tracks}",
            (w - stats_w + 15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.colors['text_white'],
            1,
            cv2.LINE_AA
        )
        
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        active_alerts: List[Dict[str, Any]],
        lstm_predictions: Dict[int, Dict[str, float]],
        involved_tracks: List[int],
        fps: float
    ) -> np.ndarray:
        """
        Run the complete drawing pipeline on a frame.
        """
        # Determine names of zones currently alerting
        alerting_zones = []
        for alert in active_alerts:
            if alert['event_type'] == 'INTRUSION':
                zname = alert['metadata'].get('zone_name')
                if zname:
                    alerting_zones.append(zname)
                    
        # 1. Draw zones
        frame = self.draw_zones(frame, alerting_zones)
        
        # 2. Draw person/object boxes
        self.draw_detections(frame, detections, involved_tracks, lstm_predictions)
        
        # 3. Draw alert banner
        self.draw_alert_banner(frame, active_alerts)
        
        # 4. Draw FPS overlay
        active_tracks_count = sum(1 for d in detections if d.track_id is not None)
        self.draw_status_overlay(frame, fps, active_tracks_count)
        
        return frame
