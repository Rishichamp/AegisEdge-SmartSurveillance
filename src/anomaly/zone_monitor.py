"""
Restricted Zone Monitor
=======================

This module monitors user-defined polygonal zones for unauthorized intrusions.

WHY POLYGON ZONES?
==================
In real-world security setups, areas are rarely simple rectangles.
They have complex boundaries - corridors, fences, gates, doorways.
By defining zones as coordinates of a polygon, we can monitor custom shapes.

HOW IT WORKS:
1. Load zones defined in configuration (polygon vertices).
2. For each detected person, compute their feet position (bottom center of bbox).
3. Perform a point-in-polygon test.
4. Track transitions (e.g. Person 5 entered zone "Server Room" at time T).
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from src.utils.geometry import point_in_polygon, bbox_center


class ZoneMonitor:
    """
    Monitors polygonal zones and tracks intrusion events for moving objects.
    """
    
    def __init__(self, zones_config: List[Dict[str, Any]]):
        """
        Args:
            zones_config: List of zone configurations, e.g.:
                          [{
                             'name': 'Restricted Server Area',
                             'polygon': [[x1, y1], [x2, y2], ...],
                             'type': 'restricted', # or 'monitored'
                             'color': [0, 0, 255]
                          }]
        """
        self.zones = []
        for z in zones_config:
            # Convert polygon coords to list of tuples of floats
            polygon = [(float(pt[0]), float(pt[1])) for pt in z['polygon']]
            self.zones.append({
                'name': z.get('name', 'Unnamed Zone'),
                'polygon': polygon,
                'type': z.get('type', 'restricted'),
                'color': z.get('color', [0, 0, 255])
            })
            
        # Keep track of active intrusions: track_id -> set of zone names they are currently in
        self.inside_tracker = {}
        
    def check_intrusions(self, person_detections: List[any]) -> List[Dict[str, Any]]:
        """
        Check if any detected persons are inside restricted zones.
        
        Args:
            person_detections: List of Detection objects with active track_ids.
            
        Returns:
            List of intrusion events triggered in this frame.
            Each event is:
            {
               'track_id': 5,
               'zone_name': 'Server Area',
               'event_type': 'entry' or 'inside',
               'type': 'restricted' or 'monitored'
            }
        """
        events = []
        current_frame_inside = {} # track_id -> list of zones they are in
        
        for det in person_detections:
            if det.track_id is None:
                continue
                
            track_id = det.track_id
            pos = bbox_center(det.bbox)
            
            for zone in self.zones:
                zone_name = zone['name']
                is_inside = point_in_polygon(pos, zone['polygon'])
                
                if is_inside:
                    if track_id not in current_frame_inside:
                        current_frame_inside[track_id] = []
                    current_frame_inside[track_id].append(zone_name)
                    
                    # Check if this is a NEW entry
                    previously_inside = self.inside_tracker.get(track_id, set())
                    
                    if zone_name not in previously_inside:
                        # TRIGGER ENTRY ALERT
                        events.append({
                            'track_id': track_id,
                            'zone_name': zone_name,
                            'event_type': 'entry',
                            'type': zone['type'],
                            'bbox': det.bbox
                        })
                    else:
                        # Log continuous presence
                        events.append({
                            'track_id': track_id,
                            'zone_name': zone_name,
                            'event_type': 'inside',
                            'type': zone['type'],
                            'bbox': det.bbox
                        })
                        
        # Identify track IDs that have exited zones
        for track_id, zones in list(self.inside_tracker.items()):
            current_zones = current_frame_inside.get(track_id, [])
            for zone_name in list(zones):
                if zone_name not in current_zones:
                    # TRIGGER EXIT EVENT (Optional logging)
                    events.append({
                        'track_id': track_id,
                        'zone_name': zone_name,
                        'event_type': 'exit',
                        'type': 'normal',
                        'bbox': None
                    })
                    
        # Update our persistent intrusion tracker state
        self.inside_tracker = {
            tid: set(zones) for tid, zones in current_frame_inside.items()
        }
        
        return events
        
    def clear_track(self, track_id: int):
        """Clean track history when object is lost."""
        if track_id in self.inside_tracker:
            del self.inside_tracker[track_id]
