"""
LSTM Action/Anomaly Classifier Model
====================================

This module implements a PyTorch LSTM (Long Short-Term Memory) classifier
for sequence-based activity recognition.

WHY LSTM FOR BEHAVIOR ANALYSIS?
===============================
Standard CNNs only look at a single image frame. They cannot understand motion or time.
For example, in a static image, a person crouching near a laptop looks normal.
But in a video, the sequence "stand -> crouch -> pick up laptop -> run" is theft.

An LSTM is a Recurrent Neural Network (RNN) designed to process sequences of data.
It maintains a internal "memory state" that gets updated with each video frame,
allowing it to learn temporal dependencies (i.e. how actions unfold over time).

TENSOR SHAPES IN RNNs (For Novices):
===================================
Input shape:  (batch_size, sequence_length, feature_dimension)
  - batch_size: number of sequences processed in parallel
  - sequence_length: number of video frames in the window (e.g. 16 frames)
  - feature_dimension: size of the feature vector extracted per frame (e.g. 576)

Output shape: (batch_size, num_classes)
  - Logits/probabilities for each class (normal, fight, theft, intrusion)
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Two-layer LSTM network with fully connected classification layers.
    Processes sequences of CNN feature vectors.
    """
    
    def __init__(
        self,
        feature_dim: int = 576,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout_prob: float = 0.3
    ):
        """
        Args:
            feature_dim: Dimension of features extracted from each crop (e.g. 576).
            hidden_size: Size of the LSTM hidden states (internal memory size).
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of target action categories.
            dropout_prob: Dropout probability to reduce overfitting.
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # --- LSTM Layer ---
        # batch_first=True makes input/output tensors have shape (batch, seq_len, feature_dim)
        # instead of the traditional (seq_len, batch, feature_dim).
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        # --- Dropout Layer ---
        # Randomly zeroes out some elements during training. Helps generalize and prevents
        # the model from memorizing the training data.
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # --- Classification Head ---
        # Maps the final LSTM hidden state to our action class predictions.
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        # x shape: (batch, seq_len, feature_dim)
        
        # Initialize hidden state (h0) and cell state (c0) with zeros.
        # Shape of h0 and c0: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Pass through the LSTM layer
        # out shape: (batch, seq_len, hidden_size)
        # (hn, cn) are the final hidden/cell states for each layer.
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We only need the LSTM output at the final timestep of the sequence.
        # out[:, -1, :] shape: (batch, hidden_size)
        last_step_out = out[:, -1, :]
        
        # Apply dropout to the hidden output
        last_step_out = self.dropout(last_step_out)
        
        # Pass through the fully connected layers to get raw logits
        # logits shape: (batch, num_classes)
        logits = self.fc(last_step_out)
        
        return logits
