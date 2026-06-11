"""
Theft Detector
==============

This module detects theft-like events in real-time.

WHY THEFT DETECTION?
====================
Theft in retail or offices often involves:
1. A person approaching a high-value object (laptop, purse, backpack).
2. The object disappearing from the scene (being picked up or hidden).
3. The person walking away shortly after the disappearance.

We implement this detection by:
- Storing a list of "monitored stationary objects" detected in the scene.
- Tracking person locations relative to these objects.
- Triggering an alert if a monitored object disappears while a person is in its immediate proximity.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.spatial import distance


class TheftDetector:
    """
    Detects theft-like scenarios: specifically, when a monitored item disappears
    while a person is nearby.
    """
    
    def __init__(
        self,
        proximity_threshold: float = 120.0,  # Max pixels distance to associate person & object
        monitored_classes: Optional[List[str]] = None,
        disappearance_frames: int = 15      # Object must be missing for 0.5s before flagging
    ):
        self.proximity_threshold = proximity_threshold
        # Default target classes for theft
        self.monitored_classes = monitored_classes or ["laptop", "cell phone", "backpack", "handbag", "suitcase"]
        self.disappearance_frames = disappearance_frames
        
        # Track active stationary items: class_id/track_id -> {bbox, center, frames_missing, last_seen_near_person_id}
        self.stationary_items = {}
        
    def analyze(
        self,
        all_detections: List[any],
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Analyze current frame detections for theft.
        
        Args:
            all_detections: List of all Detection objects in current frame.
            frame_idx: Index of current frame.
            
        Returns:
            List of theft alerts triggered.
        """
        alerts = []
        
        # Separate people and candidate theft items
        persons = []
        items_detected = []
        
        for det in all_detections:
            if det.class_name == "person":
                persons.append(det)
            elif det.class_name in self.monitored_classes:
                items_detected.append(det)
                
        # Update stationary items tracker
        current_detected_keys = set()
        
        # Process currently seen items
        for item in items_detected:
            # We construct a unique key using position and class name if track ID is not available
            item_key = f"{item.class_name}_{int(item.bbox[0]/10)}_{int(item.bbox[1]/10)}"
            current_detected_keys.add(item_key)
            
            # Find center
            x1, y1, x2, y2 = item.bbox[:4]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            # Find closest person
            closest_person_id = None
            min_dist = float('inf')
            
            for person in persons:
                px1, py1, px2, py2 = person.bbox[:4]
                p_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
                dist = distance.euclidean(center, p_center)
                
                if dist < min_dist and dist <= self.proximity_threshold:
                    min_dist = dist
                    closest_person_id = person.track_id
                    
            if item_key not in self.stationary_items:
                # Add new monitored item
                self.stationary_items[item_key] = {
                    'class_name': item.class_name,
                    'bbox': item.bbox,
                    'center': center,
                    'frames_missing': 0,
                    'last_seen_near_person': closest_person_id,
                    'frame_added': frame_idx
                }
            else:
                # Update item stats
                self.stationary_items[item_key]['bbox'] = item.bbox
                self.stationary_items[item_key]['center'] = center
                self.stationary_items[item_key]['frames_missing'] = 0
                if closest_person_id is not None:
                    self.stationary_items[item_key]['last_seen_near_person'] = closest_person_id
                    
        # Check for missing items (possible thefts)
        for key, item_data in list(self.stationary_items.items()):
            if key not in current_detected_keys:
                item_data['frames_missing'] += 1
                
                # Check if missing long enough to confirm theft (filters transient occlusion/noise)
                if item_data['frames_missing'] == self.disappearance_frames:
                    # Item disappeared! Trigger alert if a person was recently near it.
                    thief_id = item_data['last_seen_near_person']
                    
                    alerts.append({
                        'event_type': 'theft',
                        'item_class': item_data['class_name'],
                        'thief_track_id': thief_id,
                        'bbox': item_data['bbox'],
                        'confidence': 0.75 if thief_id is not None else 0.50
                    })
                    
                    # Remove from active tracking to avoid double alerts
                    del self.stationary_items[key]
            else:
                # Reset missing counter if seen again before confirmation threshold
                item_data['frames_missing'] = 0
                
        # Clean up stale items that have been missing for a very long time
        for key, item_data in list(self.stationary_items.items()):
            if item_data['frames_missing'] > 150: # 5 seconds
                del self.stationary_items[key]
                
        return alerts
