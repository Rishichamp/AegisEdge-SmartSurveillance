"""
Alert Engine
============

This module handles deduplication, priority filtering, and cooldowns for
triggered alerts.

WHY AN ALERT ENGINE?
====================
If a fight occurs, the detector will flag it on EVERY frame (30 times a second).
If we directly email, text, or store each flag, we will flood the user with alerts.

The Alert Engine prevents flooding by maintaining a "cooldown period" (e.g. 30 seconds)
per event category. If a "FIGHT_DETECTED" alert is active, another "FIGHT_DETECTED"
alert will be blocked until the cooldown expires.
"""

import time
from typing import Dict, Any, List, Optional
import os


class AlertEngine:
    """
    Orchestrates alert state, deduplication, and severity mapping.
    """
    
    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown_seconds = cooldown_seconds
        
        # Maps event_type -> timestamp of last triggered alert
        self.last_triggered = {}
        
        # History of triggered alerts
        self.history = []
        
    def trigger(
        self,
        event_type: str,
        severity: str,
        confidence: float,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to trigger an alert.
        
        Args:
            event_type: Type of event (e.g., 'FIGHT_DETECTED', 'INTRUSION')
            severity: Severity level ('INFO', 'WARNING', 'CRITICAL')
            confidence: Confidence score (0.0 to 1.0)
            description: Plain-text description of the event.
            metadata: Extra key-value pairs.
            
        Returns:
            Alert dictionary if the alert was successfully triggered (not throttled),
            None if it was suppressed by the cooldown.
        """
        now = time.time()
        last_time = self.last_triggered.get(event_type, 0.0)
        
        # Check cooldown
        if now - last_time < self.cooldown_seconds:
            # Alert is suppressed
            return None
            
        # Trigger the alert
        self.last_triggered[event_type] = now
        
        alert_event = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            'timestamp_unix': now,
            'event_type': event_type,
            'severity': severity,
            'confidence': float(confidence),
            'description': description,
            'metadata': metadata or {}
        }
        
        self.history.append(alert_event)
        
        # Keep history capped at 1000 items
        if len(self.history) > 1000:
            self.history.pop(0)
            
        print(f"[{severity}] ALERT TRIGGERED: {description} (Conf: {confidence:.2f})")
        return alert_event
        
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.history[-limit:]
