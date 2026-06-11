# System Components — AegisEdge Smart Surveillance
===================================================

This document details the core components of the AegisEdge Smart Surveillance system. It explains the class structures, algorithms, and options for each module.

---

## 1. Core Utilities (`src/utils/`)

### 1.1. `VideoStream` ([video_stream.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/utils/video_stream.py))
* **Purpose:** Provides thread-safe, buffered video frame capture from webcams, RTSP streams, and video files.
* **Why it matters:** OpenCV's default `cv2.VideoCapture().read()` blocks the execution thread. On network cameras, it buffers old frames. `VideoStream` decouples capturing from processing.
* **Key Configurations:**
  * `source`: Camera index (`0`) or RTSP URL (`rtsp://...`) or file path.
  * `width`/`height`: Resizes frames immediately to optimize memory layout.
  * `buffer_size`: Size of the frame queue (typically `2` to keep latency low).

### 1.2. `Geometry` ([geometry.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/utils/geometry.py))
* **Purpose:** Encapsulates geometrical mathematics (intersection-over-union, point-in-polygon tests).
* **Algorithm:**
  * Uses the **Ray-Casting Algorithm** to test if a 2D point (representing a person's feet) is inside a polygonal restricted zone defined by vertices $(x_i, y_i)$.
  * Formulates boundary intersections to count crossings; odd crossings mean the point is inside the polygon.

---

## 2. Computer Vision & Detection (`src/detection/`)

### 2.1. `ObjectDetector` ([detector.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/detection/detector.py))
* **Purpose:** Wrapper for Ultralytics YOLOv8 inference. Supports standard object detection and object tracking.
* **Method:**
  * `detect_and_track()`: Invokes YOLOv8's built-in ByteTrack module. It tracks objects across frames and manages Kalman-filter states, returning `Detection` dataclasses with persistent tracking IDs.

### 2.2. `TrackHistoryManager` ([tracker.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/detection/tracker.py))
* **Purpose:** Stores coordinate histories for active tracking IDs to build path trails and calculate velocities.
* **Math:**
  * **Velocity Vector:** Calculates displacement $\Delta x, \Delta y$ over a rolling sliding window of frames:
    $$dx = \frac{x_{t} - x_{t-w}}{w}, \quad dy = \frac{y_{t} - y_{t-w}}{w}$$
  * **Speed:** Computes Euclidean velocity magnitude:
    $$\text{Speed} = \sqrt{dx^2 + dy^2}$$

---

## 3. Deep Learning Posture Sequence Models (`src/models/` & `src/anomaly/`)

### 3.1. `ROIFeatureExtractor` ([feature_extractor.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/anomaly/feature_extractor.py))
* **Purpose:** Crops a region of interest (person box) and extracts a fixed-size deep representation.
* **Architecture:**
  * Pretrained **MobileNetV3-Small** backbone.
  * Discards the final fully-connected classifier layers.
  * Runs global average pooling to output a **576-dimensional feature vector** capturing poses, postures, and features.

### 3.2. `LSTMClassifier` ([lstm_classifier.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/models/lstm_classifier.py))
* **Purpose:** PyTorch Recurrent Neural Network (RNN) that processes temporal feature sequences.
* **Architecture:**
  * 2 LSTM layers with hidden state size `256` and dropout of `0.3`.
  * Fully connected layers map the final timestep's output to class logits representing four behavior categories: `[normal, fight, theft, intrusion]`.

### 3.3. `TemporalAnomalyPredictor` ([temporal_model.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/anomaly/temporal_model.py))
* **Purpose:** Maintains a sliding queue buffer (`collections.deque`) of size 16 for each active tracking ID.
* **Fallback Behavior:**
  * If weights are not loaded (e.g. out-of-the-box run), it computes variance over sequence features. High variance flags chaotic movements (fights), medium variance flags rapid picking actions (theft), and low variance represents normal walking, providing a clean demo without data overhead.

---

## 4. Threat Analytic Engines (`src/anomaly/`)

### 4.1. `FightDetector` ([fight_detector.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/anomaly/fight_detector.py))
* **Purpose:** Hybrid physics-based and ML-based altercation classifier.
* **Logic:**
  1. Checks if at least two people are in close proximity (Euclidean distance threshold: e.g. 150px).
  2. Computes **Dense Optical Flow (Farneback Algorithm)** on consecutive frames to measure overall scene kinetic energy.
  3. Fuses proximity, motion energy, and the LSTM sequence score. If the combined score exceeds `0.55`, it flags a fight.

### 4.2. `ZoneMonitor` ([zone_monitor.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/anomaly/zone_monitor.py))
* **Purpose:** Tests person foot coordinates (bottom-center of bounding boxes) against user-defined restricted zones.
* **State Machine:**
  * Tracks enter/inside/exit state transitions to trigger alerts only when an object crosses the threshold.

### 4.3. `TheftDetector` ([theft_detector.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/anomaly/theft_detector.py))
* **Purpose:** Detects retail shoplifting or office asset theft.
* **State Machine:**
  1. Monitors key object classes (laptops, bags, cell phones) and associates them with nearby people.
  2. If a monitored object disappears, its missing counter increments.
  3. If missing for more than `15` frames (occlusion filter), it triggers a theft warning, associating the incident with the last tracked person seen near the object.

---

## 5. Alerts & Output Logging (`src/alerting/`)

### 5.1. `AlertEngine` ([alert_engine.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/alerting/alert_engine.py))
* **Purpose:** Applies cooldown throttles to prevent spamming notifications during persistent threats.
* **Logic:**
  * Stores timestamps of triggered alerts by type. Suppresses alerts of the same type within the cooldown window (e.g. 30 seconds).

### 5.2. `EventLogger` ([logger.py](file:///C:/Users/Rishi%20Singh/.gemini/antigravity/scratch/smart-surveillance-system/src/alerting/logger.py))
* **Purpose:** Writes events to disk and manages snapshot captures.
* **Output Format:**
  * **JSON Lines (`.jsonl`):** Log database where each line is a structured JSON alert event, perfect for indexing by log servers (ELK stack).
  * **Images (`.jpg`):** Saves the annotated video frame containing the threat to disk for inspection.
