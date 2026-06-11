"""
ROI Feature Extractor (MobileNetV3)
==================================

This module extracts low-dimensional visual feature vectors from cropped regions
of interest (ROIs) - specifically, bounding boxes containing detected persons.

WHY MOBILENETV3?
================
We need to extract high-level visual features (pose, posture, appearance) from
each person crop to feed into our LSTM temporal classifier.

If we use a heavy network like ResNet50, inference will be extremely slow.
MobileNetV3-Small is designed specifically for mobile and edge platforms. It is
extremely fast and light, yet provides a rich 576-dimensional feature vector
that captures the essential shape and action details of the person crop.

HOW IT WORKS:
1. Receive bounding box coordinates [x1, y1, x2, y2]
2. Crop the person from the frame
3. Resize the crop to 224x224 (expected input size for MobileNetV3)
4. Normalize the crop (mean/std subtraction)
5. Pass through the backbone network (stopping before the final classification head)
6. Output a 576-dimensional feature vector
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Optional


class ROIFeatureExtractor:
    """
    Extracts visual features from image crops using a pretrained MobileNetV3.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Device to run extractor on ("cpu", "cuda", etc.)
        """
        self.device = torch.device(device)
        print(f"[EXTRACTOR] Initializing MobileNetV3-Small on {device}")
        
        # Load pre-trained MobileNetV3-Small
        # weights=models.MobileNet_V3_Small_Weights.DEFAULT loads the latest weights
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # We only want the feature extractor backbone, NOT the final classifier head.
        # MobileNetV3's backbone is stored in `.features`.
        # Passing an input through `.features` followed by global average pooling
        # yields a 576-dimensional feature vector.
        self.features = self.model.features
        self.features.eval()  # Set model to evaluation mode
        self.features.to(self.device)
        
        # Global Average Pooling layer to convert 3D feature maps to 1D vectors
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # Standard ImageNet pre-processing transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    @torch.no_grad()
    def extract_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature vector from a cropped frame.
        
        Args:
            crop: BGR NumPy array representing the image crop.
            
        Returns:
            NumPy array of shape (576,) representing extracted features,
            or None if the crop is invalid.
        """
        if crop is None or crop.size == 0:
            return None
            
        # OpenCV reads in BGR, PyTorch models expect RGB. Convert.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy array to PIL Image for torchvision transforms
        pil_img = Image.fromarray(crop_rgb)
        
        # Preprocess and add batch dimension: shape becomes (1, 3, 224, 224)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Forward pass through features backbone
        # Output shape: (1, 576, 7, 7)
        feat_map = self.features(tensor_img)
        
        # Apply global pooling: shape becomes (1, 576, 1, 1)
        pooled = self.pool(feat_map)
        
        # Flatten to shape (576,)
        feat_vector = torch.flatten(pooled).cpu().numpy()
        
        return feat_vector

    def extract_from_frame(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop person from a full frame and extract their features.
        
        Args:
            frame: Full BGR video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Clip coordinates to frame boundaries to prevent crash
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = frame[y1:y2, x1:x2]
        return self.extract_features(crop)
