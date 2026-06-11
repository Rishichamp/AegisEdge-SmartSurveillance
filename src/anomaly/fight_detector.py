"""
Fight / Violence Detector
=========================

This module implements a hybrid approach to real-time fight and violence detection.

WHY HYBRID DETECTION?
=====================
Pure deep learning models (like 3D CNNs) can be computationally expensive and may
suffer from false positives (e.g. people hugging or waving hands).

To build a reliable surveillance system, we combine:
1. **Proximity Checks (Rule-based):** Fights require 2+ people in close proximity.
2. **Motion Analysis (Computer Vision):** Fights exhibit high-energy, chaotic motion.
   We measure frame-to-frame pixel changes (optical flow or frame differencing).
3. **LSTM Classification (Temporal Deep Learning):** Classifies the pattern of actions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from scipy.spatial import distance


class FightDetector:
    """
    Detects physical fights or violence based on physical proximity, motion levels,
    and temporal model action classification scores.
    """
    
    def __init__(
        self,
        proximity_threshold: float = 150.0, # Pixels
        motion_threshold: float = 30.0,      # Optical flow motion energy
        min_persons: int = 2,
        lstm_weight: float = 0.6,
        rules_weight: float = 0.4
    ):
        self.proximity_threshold = proximity_threshold
        self.motion_threshold = motion_threshold
        self.min_persons = min_persons
        self.lstm_weight = lstm_weight
        self.rules_weight = rules_weight
        
        # Buffer to keep track of the previous frame's gray image (for motion analysis)
        self.prev_gray = None
        
    def calculate_motion_energy(self, frame: np.ndarray) -> float:
        """
        Calculate motion energy between the current frame and the previous frame
        using Farneback Optical Flow or average magnitude.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
            
        # Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.prev_gray,
            next=gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Compute magnitude of vectors: sqrt(dx^2 + dy^2)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = float(np.mean(magnitude))
        
        self.prev_gray = gray
        return avg_motion
        
    def check_proximity(self, person_bboxes: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        """
        Find pairs of people who are close to each other.
        
        Returns:
            List of tuples (index1, index2, distance_pixels)
        """
        if len(person_bboxes) < self.min_persons:
            return []
            
        # Extract centers of all bounding boxes
        centers = []
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox[:4]
            centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            
        centers = np.array(centers)
        
        # Compute pairwise distance matrix
        dist_matrix = distance.cdist(centers, centers, 'euclidean')
        
        close_pairs = []
        n = len(person_bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_matrix[i, j]
                if dist <= self.proximity_threshold:
                    close_pairs.append((i, j, float(dist)))
                    
        return close_pairs
        
    def analyze(
        self,
        frame: np.ndarray,
        person_detections: List[any],
        lstm_predictions: Dict[int, Dict[str, float]]
    ) -> Tuple[bool, float, List[int]]:
        """
        Analyze the current frame for fight/violence.
        
        Args:
            frame: Current BGR image
            person_detections: List of Detection objects
            lstm_predictions: Dictionary mapping track_id -> action probabilities
            
        Returns:
            - fight_detected: True/False
            - confidence: Combined confidence score (0.0 to 1.0)
            - involved_tracks: List of track IDs involved
        """
        if len(person_detections) < self.min_persons:
            return False, 0.0, []
            
        # 1. Calculate general motion in frame
        motion_energy = self.calculate_motion_energy(frame)
        motion_score = min(1.0, motion_energy / self.motion_threshold)
        
        # 2. Check proximity
        bboxes = [det.bbox for det in person_detections]
        close_pairs = self.check_proximity(bboxes)
        
        if not close_pairs:
            return False, 0.0, []
            
        # 3. Combine with LSTM predictions for involved tracks
        involved_tracks = []
        lstm_fight_scores = []
        
        # For each pair in close proximity, check if LSTM flags "fight" behavior
        for idx1, idx2, dist in close_pairs:
            track1 = person_detections[idx1].track_id
            track2 = person_detections[idx2].track_id
            
            # Keep track of involved IDs
            if track1 is not None:
                involved_tracks.append(track1)
                if track1 in lstm_predictions:
                    lstm_fight_scores.append(lstm_predictions[track1].get("fight", 0.0))
            if track2 is not None:
                involved_tracks.append(track2)
                if track2 in lstm_predictions:
                    lstm_fight_scores.append(lstm_predictions[track2].get("fight", 0.0))
                    
        # Remove duplicates
        involved_tracks = list(set(involved_tracks))
        
        # Calculate average LSTM fight score for involved parties
        avg_lstm_score = float(np.mean(lstm_fight_scores)) if lstm_fight_scores else 0.0
        
        # Rules-based score: close proximity + high motion
        rules_score = 0.5 * motion_score + 0.5 * (1.0 - (min(dist, self.proximity_threshold) / self.proximity_threshold))
        
        # Fuse scores
        combined_score = (self.lstm_weight * avg_lstm_score) + (self.rules_weight * rules_score)
        
        # We flag a fight if the combined score exceeds 0.55
        is_fight = combined_score >= 0.55
        
        return is_fight, combined_score, involved_tracks
