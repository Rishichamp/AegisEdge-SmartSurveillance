"""
Temporal Anomaly Predictor
==========================

This module manages the temporal sequence buffering per track ID and feeds
these sequences to the LSTM action classifier.

WHY SEQUENCE BUFFERING?
=======================
We process video frame-by-frame. To run a temporal sequence model (LSTM),
we need to gather the features of a specific person over a sliding window
of frames (e.g. 16 frames).

Since multiple people can appear in the camera view simultaneously, we must
maintain a separate sliding buffer of feature vectors for each active track ID.
"""

import collections
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.models.lstm_classifier import LSTMClassifier


class TemporalAnomalyPredictor:
    """
    Manages temporal buffers of features for tracked persons and runs the
    LSTM action classifier for anomaly prediction.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        architecture: str = "lstm",
        sequence_length: int = 16,
        feature_dim: int = 576,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        device: str = "cpu"
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # Initialize the PyTorch model
        self.model = LSTMClassifier(
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Load weights if available
        self.has_weights = False
        if weights_path is not None:
            try:
                print(f"[TEMPORAL] Loading weights from: {weights_path}")
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.has_weights = True
            except Exception as e:
                print(f"[TEMPORAL] Warning: Could not load weights: {e}. Running with random/demo initialization.")
                
        self.model.to(self.device)
        self.model.eval()
        
        # Store for buffers: track_id -> deque of feature vectors (size = sequence_length)
        self.feature_buffers = {}
        
        # Labels mapping
        self.class_labels = {
            0: "normal",
            1: "fight",
            2: "theft",
            3: "intrusion"
        }
        
    def add_features(self, track_id: int, features: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Add a new feature vector to the sliding window buffer for a specific track ID.
        If the buffer reaches the required sequence length, run inference and return predictions.
        
        Args:
            track_id: Persistent ID of the tracked person.
            features: 576-dim feature vector.
            
        Returns:
            Dictionary of class probabilities (e.g. {'normal': 0.8, 'fight': 0.1, ...})
            or None if the buffer is not yet full.
        """
        if track_id not in self.feature_buffers:
            self.feature_buffers[track_id] = collections.deque(maxlen=self.sequence_length)
            
        self.feature_buffers[track_id].append(features)
        
        # Check if we have enough temporal context
        if len(self.feature_buffers[track_id]) == self.sequence_length:
            return self._predict(track_id)
            
        return None
        
    def clear_track(self, track_id: int):
        """Clear the feature buffer when a track is lost."""
        if track_id in self.feature_buffers:
            del self.feature_buffers[track_id]
            
    @torch.no_grad()
    def _predict(self, track_id: int) -> Dict[str, float]:
        """
        Perform forward pass on the sequence buffer for a track ID.
        """
        if not self.has_weights:
            # Fallback Demo Mode: Returns synthetic/simulated behavior scores
            # based on features statistics so the system demonstrates nicely without training data.
            feats = np.array(self.feature_buffers[track_id])
            # Calculate mock variances to simulate motion-based anomalies
            var = np.var(feats, axis=0).mean()
            
            # Simple heuristic mapping for demo
            if var > 0.05:  # High variance (fast flailing motion) -> Mock fight
                scores = [0.1, 0.7, 0.1, 0.1]
            elif var > 0.02 and np.mean(feats) > 0.1:  # Medium variance, specific posture -> Mock theft
                scores = [0.2, 0.1, 0.6, 0.1]
            else:
                scores = [0.9, 0.02, 0.04, 0.04]
                
            # Softmax normalize scores
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            
            return {self.class_labels[i]: float(probs[i]) for i in range(self.num_classes)}
            
        # Convert sequence buffer to PyTorch tensor (shape: 1, seq_len, feat_dim)
        seq = np.array(self.feature_buffers[track_id], dtype=np.float32)
        tensor_seq = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        
        logits = self.model(tensor_seq)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        return {self.class_labels[i]: float(probabilities[i]) for i in range(self.num_classes)}
