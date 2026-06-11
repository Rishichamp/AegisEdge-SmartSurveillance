# Placement Interview Preparation Guide
========================================

This guide prepares you to present AegisEdge in placement interviews for Computer Vision, Edge AI, or Robotics software roles.

---

## 1. The 60-Second Elevator Pitch

"AegisEdge is a real-time, multi-threaded Smart Surveillance Edge AI system that processes video streams (webcam, RTSP, or local files) to track multiple objects and detect behavior anomalies like physical fights, restricted zone intrusions, and retail theft. 

The pipeline uses YOLOv8 and ByteTrack to track people and assets. It routes cropped person boxes through a MobileNetV3 feature extractor, feeding a sequence of visual postures into a PyTorch LSTM classifier to model temporal behaviors over time. 

Alert events are processed through an engine that handles deduplication and cooldowns, logged to an append-only JSONL database with saved screenshots, and streamed via WebSockets to a sleek glassmorphic Flask dashboard in real time. It is fully containerized and optimized for edge devices like NVIDIA Jetson using ONNX and TensorRT."

---

## 2. Core Technical Questions & Answers

### Q1: Why did you choose YOLOv8 over older models like YOLOv5 or heavy architectures like Detectron2?
* **Answer:** "YOLOv8 represents the state-of-the-art in real-time object detection. It is anchor-free, meaning it predicts boxes directly from centers, which improves speed and box precision. Unlike Detectron2, which runs heavier Two-Stage Mask R-CNN frameworks and is slow on edge hardware, YOLOv8 is a One-Stage detector that runs at high frame rates. It also provides native tracking support (ByteTrack/BoT-SORT), making multi-object tracking a single-step operation in the code."

### Q2: How does the ByteTrack algorithm work? Why not use standard SORT?
* **Answer:** "Standard SORT (Simple Online and Realtime Tracking) throws away detections below a certain confidence threshold. However, during occlusion (e.g. a person walking behind a pillar), detector confidence drops, causing track fragmentation and ID switches. 
ByteTrack solves this by keeping all detections. It first matches high-confidence boxes using Kalman filters and IoU overlap. Then, for the remaining unmatched tracks, it attempts a second association step using the *low-confidence* boxes. This lets the system track people through partial occlusions and camera noise without switching their track IDs."

### Q3: Explain why you used a MobileNetV3 + LSTM hybrid model instead of an end-to-end 3D Convolutional Network (3D CNN).
* **Answer:** "End-to-end 3D CNNs (like I3D or SlowFast) process spatial and temporal dimensions simultaneously. They are highly accurate but computationally expensive, requiring massive floating-point operations (FLOPs) that would freeze a CPU or Jetson Nano. 
My hybrid design decouples space and time. MobileNetV3-Small is an extremely light CNN designed for mobile CPUs. It acts as a spatial feature extractor, producing a 576-dimensional vector for person crops. The LSTM (Long Short-Term Memory) is a Recurrent Neural Network (RNN) that only models the temporal aspect, mapping the 16-frame sequence history to behavior logits. This split decreases computation requirements by over 70%, allowing edge execution."

### Q4: How does the web dashboard stream live video without locking up the Flask server?
* **Answer:** "Flask's standard request-response model blocks when a client requests a video feed. I resolved this by running the Flask server and the core AI processing pipeline in separate threads. The main processing pipeline updates a thread-safe frame buffer. Flask streams this buffer as an MJPEG multipart response (`multipart/x-mixed-replace`) in a separate generator function. At the same time, hardware metrics and event cards are pushed asynchronously to the browser using **Socket.io WebSockets**, preventing network bottlenecks and UI freezes."

### Q5: How did you implement restricted zone monitor math?
* **Answer:** "I implemented the Ray-Casting (or Even-Odd) Point-in-Polygon algorithm. First, I determine the person's feet coordinate as the bottom center of the bounding box: $(x_{\text{center}}, y_{\text{bottom}})$. Then, I project a horizontal ray to the right from this coordinate. I calculate the intersections between this ray and all edges of the restricted zone polygon. If the number of intersections is odd, the person is inside the zone. If it's even, they are outside. By using feet instead of the box center, we prevent false alarms from shadows or overhead objects."

---

## 3. Scenario & Problem Solving Questions

### Q1: "Your system is lagging and dropping frames on a Jetson Nano. What steps do you take to optimize it?"
* **Answer:**
  1. "Verify if bottlenecking is in the CPU or GPU using command line monitors like `tegrastats`."
  2. "If the CPU is bottlenecked, increase the `frame_skip` setting (e.g., process every 2nd or 3rd frame), which reduces tracking and feature extraction loads."
  3. "If the GPU is bottlenecked, export the PyTorch models to ONNX and compile them into optimized **TensorRT engine files (`.engine`)** using FP16 quantization, which leverages GPU Tensor Cores."
  4. "Reduce input resolution from `1080p` to `640x480`. Bounding box regression is faster on smaller inputs, and YOLOv8-nano is highly accurate at this resolution."

### Q2: "How do you handle false alarms in fight detection (e.g. two people hugging or shaking hands)?"
* **Answer:** "This is a common issue with spatial proximity models. I mitigated this by implementing a multi-modal fusion logic:
  - Hugging is slow. Fights are fast. I use **Farneback Dense Optical Flow** to calculate the kinetic motion energy between frames. A fight is flagged only if proximity is close AND average optical flow motion exceeds a threshold (e.g., 30 pixels).
  - Additionally, our LSTM model analyzes a temporal window of 16 frames. A hug has a steady, unified visual signature, whereas punches or struggles have chaotic, oscillating posture sequences. The LSTM filters out steady-state poses."
