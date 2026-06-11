"""
Geometry Utilities
==================

This module provides geometric operations needed for zone-based
intrusion detection. The key operation is checking whether a point
(person's position) falls inside a polygon (restricted zone).
"""

import cv2
import numpy as np
from typing import List, Tuple


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon.
    
    Uses OpenCV's pointPolygonTest.
    
    Args:
        point: (x, y) coordinates of the point.
        polygon: List of (x, y) coordinates defining the polygon.
    
    Returns:
        True if the point is inside or on the boundary of the polygon.
    """
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False)
    return result >= 0


def bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the bottom center of a bounding box.
    We use the bottom center because it represents where the person stands on the ground.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2].
    
    Returns:
        (x_center, y_bottom)
    """
    x1, y1, x2, y2 = bbox[:4]
    x_center = (x1 + x2) / 2.0
    return (float(x_center), float(y2))


def bbox_center_middle(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the true center of a bounding box.
    """
    x1, y1, x2, y2 = bbox[:4]
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    return (float(x_center), float(y_center))


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0.0 and 1.0.
    """
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0.0:
        return 0.0
        
    return intersection / union
