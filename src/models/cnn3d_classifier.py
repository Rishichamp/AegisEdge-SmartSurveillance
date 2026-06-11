"""
3D CNN Action Classifier Model
==============================

This module implements a 3D Convolutional Neural Network (CNN) for spatio-temporal
activity recognition in video clips.

WHY 3D CNN?
===========
While standard 2D CNNs learn spatial features from images, and LSTMs learn temporal
relationships from sequences, a 3D CNN learns spatial and temporal features
JOINTLY. 

In a 3D Convolution, the kernels (filters) slide not only across the width (x)
and height (y) of the image, but also across the time (t) axis of consecutive frames.
This enables the network to naturally capture complex motion dynamics, like the flailing
limbs in a fight or the quick reaching movement of theft, directly from raw frames.

TENSOR SHAPES IN 3D CNNs:
=========================
Input shape: (batch_size, channels, sequence_length, height, width)
  - batch_size: number of video clips processed in parallel
  - channels: 3 for RGB image channels
  - sequence_length: number of frames in the clip (e.g. 16 frames)
  - height, width: spatial dimensions of the input frames (e.g. 112x112)
"""

import torch
import torch.nn as nn


class CNN3DClassifier(nn.Module):
    """
    Lightweight 3D CNN Action Classifier.
    Appropriate for real-time inference on systems with moderate CPU/GPU capabilities.
    """
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3, sequence_length: int = 16):
        """
        Args:
            num_classes: Number of output action classes.
            in_channels: Number of input channels (typically 3 for RGB).
            sequence_length: Number of frames per video clip.
        """
        super(CNN3DClassifier, self).__init__()
        
        # --- Feature Extractor (3D CNN Layers) ---
        # Layer 1: Conv3D -> Batch Normalization -> ReLU -> MaxPool3D
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # MaxPool3d shrinks dimensions. kernel=(1, 2, 2) keeps temporal resolution,
            # but halves spatial height and width.
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # MaxPool3d kernel=(2, 2, 2) halves temporal AND spatial dimensions
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # --- Global Average Pooling ---
        # Reduces the (batch, 128, T_out, H_out, W_out) tensor to (batch, 128, 1, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # --- Classifier Head ---
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, T, H, W)
               e.g. (batch, 3, 16, 112, 112)
        """
        # Feature extraction
        x = self.conv1(x)  # Shape: (batch, 32, 16, 56, 56)
        x = self.conv2(x)  # Shape: (batch, 64, 8, 28, 28)
        x = self.conv3(x)  # Shape: (batch, 128, 4, 14, 14)
        
        # Global pooling
        x = self.avgpool(x) # Shape: (batch, 128, 1, 1, 1)
        
        # Flatten tensor to (batch, 128)
        x = torch.flatten(x, 1)
        
        # Classification
        logits = self.fc(x) # Shape: (batch, num_classes)
        
        return logits
