# AegisEdge — Real-Time Edge AI Surveillance System

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)

AegisEdge is a production-grade, real-time computer vision surveillance system that runs on local workstations and edge hardware (NVIDIA Jetson Nano/Orin, Raspberry Pi 4/5). It processes live video streams to track multiple objects simultaneously and detect security threats — **intrusions, fights, and asset theft** — using a hybrid AI + rule-based analytics pipeline.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Tech Stack](#tech-stack)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Running the System](#running-the-system)
7. [Configuration](#configuration)
8. [How It Works](#how-it-works)
9. [Training & Edge Export](#training--edge-export)
10. [Unit Testing](#unit-testing)
11. [Troubleshooting](#troubleshooting)

---

## Key Features

- **Multi-Object Tracking** — YOLOv8 + ByteTrack with Kalman filter smoothing tracks people and assets (laptops, bags, phones) across frames, robust to partial occlusion and ID switching
- **Restricted Zone Intrusion Detection** — Custom polygonal zone definitions with ray-casting point-in-polygon math; edge-triggered state machine fires on entry transition only, not continuous presence
- **Fight & Violence Detection** — Three-signal fusion: spatial proximity gate + Farneback dense optical flow magnitude + PyTorch LSTM posture sequence classifier
- **Asset Theft Detection** — Spatial-temporal state machine detects when monitored objects disappear while a nearby person is tracked; flags the last proximate person as suspect
- **Three-Tier Threaded Pipeline** — Separate threads for video capture, AI inference, and web dashboard maintain real-time FPS without frame lag or UI freeze
- **Throttled Alert Engine** — Per-threat cooldown windows suppress duplicate alerts; events logged to append-only JSONL + annotated screenshots saved automatically
- **Glassmorphic Web Dashboard** — Real-time MJPEG video stream, inference FPS + CPU metrics, dynamic alert cards, investigation modal with suspect ID and screenshot

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | ByteTrack + Kalman Filter |
| Spatial Feature Extraction | MobileNetV3-Small (torchvision) |
| Temporal Behavior Modeling | PyTorch LSTM |
| Optical Flow | OpenCV Farneback Dense Flow |
| Zone Intrusion | Ray-Casting Point-in-Polygon |
| Web Dashboard | Flask + Flask-SocketIO + MJPEG |
| Edge Deployment | ONNX + NVIDIA TensorRT |
| Containerization | Docker (Jetson L4T + RPi ARM64) |
| Language | Python 3.8+ |

---

## System Architecture

```
Video Source (Webcam / RTSP / File)
         ↓
 Thread 1 — VideoStream          ← Double-buffered queue, drops stale frames
         ↓
 Thread 2 — SurveillancePipeline
   ├── YOLOv8 + ByteTrack        ← Detection + multi-object tracking
   ├── MobileNetV3 Extractor     ← 576-dim spatial feature per person crop
   ├── LSTM Temporal Classifier  ← 16-frame rolling window → behavior score
   └── Threat Analytics Engines
         ├── FightDetector       ← Proximity + optical flow + LSTM
         ├── ZoneMonitor         ← Ray-casting polygon intrusion
         └── TheftDetector       ← Asset disappearance state machine
         ↓
 Alert Engine                    ← Cooldown deduplication + JSONL log + screenshots
         ↓
 Thread 3 — DashboardServer      ← Flask + SocketIO, MJPEG stream, alert events
         ↓
 Browser Dashboard               ← Live video + alert cards + investigation modal
```

**Why three threads?** Deep learning inference (YOLOv8 + MobileNetV3 + LSTM) is CPU-intensive. A single-threaded design would cause video lag and UI freezes. Thread 1 always captures the newest frame; Thread 2 runs inference; Thread 3 serves the web UI — all independently, with no blocking between them.

---

## Project Structure

```
AegisEdge-SmartSurveillance/
├── src/
│   ├── pipeline.py              # Master orchestrator — wires all modules together
│   ├── config.py                # YAML config loader and accessor
│   ├── detection/
│   │   ├── detector.py          # YOLOv8 wrapper with NMS and class filtering
│   │   └── tracker.py           # ByteTrack coordinate history and ID management
│   ├── anomaly/
│   │   ├── fight_detector.py    # Proximity + optical flow + LSTM fusion
│   │   ├── zone_monitor.py      # Polygon zone definitions + ray-casting PIP
│   │   ├── theft_detector.py    # Asset disappearance state machine
│   │   ├── feature_extractor.py # MobileNetV3 ROI feature extraction
│   │   └── temporal_model.py    # LSTM sequence buffer and inference
│   ├── models/
│   │   ├── lstm_classifier.py   # PyTorch LSTM network definition
│   │   └── cnn3d_classifier.py  # 3D CNN alternative architecture
│   ├── alerting/
│   │   ├── alert_engine.py      # Cooldown deduplication + event dispatch
│   │   └── logger.py            # JSONL append-only event logger
│   ├── visualization/
│   │   ├── annotator.py         # Frame drawing (boxes, zones, labels, trails)
│   │   └── dashboard.py         # Flask + SocketIO server
│   └── utils/
│       ├── video_stream.py      # Thread-safe double-buffered video capture
│       ├── geometry.py          # Polygon math, Euclidean distance utilities
│       └── threading_utils.py   # Worker thread base classes
├── scripts/
│   ├── run_demo.py              # Built-in demo, no webcam needed
│   ├── run_surveillance.py      # Main entry point with CLI argument overrides
│   ├── train_anomaly_model.py   # LSTM training pipeline
│   └── export_model.py          # ONNX + TensorRT export
├── dashboard/                   # Web UI (HTML / CSS / JS)
│   ├── index.html
│   ├── style.css                # Glassmorphism dark-mode theme
│   └── app.js                   # Socket.io WebSocket client
├── config/
│   ├── default_config.yaml      # Desktop / laptop settings
│   ├── jetson_config.yaml       # Jetson Nano/Orin (CUDA + TensorRT)
│   └── rpi_config.yaml          # Raspberry Pi (CPU, frame-skipping)
├── docker/
│   ├── Dockerfile.jetson        # NVIDIA L4T base image
│   └── Dockerfile.rpi           # ARM64 slim Python image
├── tests/                       # Unit tests (pytest / unittest)
├── models/                      # Model weight files (.pt / .onnx) — gitignored
├── data/                        # Video files and zone configs — gitignored
├── logs/                        # Auto-generated alert logs and screenshots
├── requirements.txt
└── setup.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher (`python --version` to check)
- Git (`git --version` to check)
- C++ Build Tools (needed to compile `scipy` / `psutil` if no binary wheel is available):
  - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → select "Desktop development with C++"
  - **macOS:** Run `xcode-select --install`
  - **Linux:** Run `sudo apt update && sudo apt install build-essential python3-dev`

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AegisEdge-SmartSurveillance.git
cd AegisEdge-SmartSurveillance
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the virtual environment

```bash
# Windows (PowerShell)
# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

Your terminal prompt will show `(venv)` when activated.

### Step 4 — Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install psutil
```

This downloads approximately 300 MB of packages including PyTorch, YOLOv8, and OpenCV.

---

## Running the System

### Option A — Built-in Demo (no webcam needed)

Simulates three actors triggering all three alert types:

```bash
python scripts/run_demo.py
```

Open **http://localhost:5000** in your browser and watch:

| Time | Event |
|---|---|
| 0s – 4s | Actor #2 enters the red restricted polygon → **Intrusion Alert** |
| 8s – 10s | Actors #3 & #4 approach with chaotic motion → **Fight Alert** |
| 17s – 21s | Actor #5 picks up a laptop and it disappears → **Theft Alert** |

Click any alert card to open the investigation modal with the captured screenshot and suspect ID.

### Option B — Live webcam

```bash
python scripts/run_surveillance.py --source 0
# Try --source 1 or --source 2 if you have multiple cameras
```

### Option C — Video file

```bash
python scripts/run_surveillance.py --source "path/to/your/video.mp4"
```

### Option D — Custom config

```bash
python scripts/run_surveillance.py --config config/jetson_config.yaml
```

---

## Configuration

All behavior is controlled via `config/default_config.yaml`. Key settings:

| Category | Key | Default | Description |
|---|---|---|---|
| Video | `video.source` | `0` | Camera index, file path, or RTSP URL |
| | `video.width` / `height` | `640` / `480` | Resolution — smaller is faster on CPU |
| Detection | `detection.model` | `yolov8n.pt` | Model size: n (fast) → s → m → l → x (accurate) |
| | `detection.device` | `cpu` | `cpu` or `cuda` |
| | `detection.confidence_threshold` | `0.5` | Minimum detection confidence |
| Fight | `anomaly.fight_detection.proximity_threshold` | `150` | Max pixel distance between persons to check |
| | `anomaly.fight_detection.motion_threshold` | `30.0` | Optical flow magnitude to confirm fight motion |
| Theft | `anomaly.theft_detection.monitored_objects` | `["laptop", "cell phone"]` | Asset classes to watch |
| Zones | `zones[].polygon` | `[[x,y], ...]` | Vertex coordinates of restricted polygons |
| Alerting | `alerting.cooldown_seconds` | `30` | Seconds before a duplicate alert can fire |
| Dashboard | `dashboard.port` | `5000` | Web server port |

**Example — adding a custom restricted zone:**

```yaml
zones:
  - name: "Server Room Door"
    polygon: [[50, 200], [300, 200], [300, 480], [50, 480]]
    type: "restricted"
    color: [0, 0, 255]   # Red in BGR
```

---

## How It Works

### Fight Detection — Three-Signal Fusion

A fight alert fires only when **all three conditions** are satisfied simultaneously, keeping false positives extremely low:

1. **Proximity gate** — Two or more persons are within `proximity_threshold` pixels (Euclidean distance between bounding box centers)
2. **Optical flow** — Farneback dense flow magnitude averaged over the region between persons exceeds `motion_threshold`
3. **LSTM score** — The 16-frame posture sequence classifier returns `P(fight) > confidence_threshold`

### Zone Intrusion — Ray-Casting Point-in-Polygon

To check if a person's feet coordinate is inside a polygon, a horizontal ray is cast from that point and polygon edge crossings are counted. An odd count means the point is inside:

```
inside = (crossings % 2 == 1)
```

Only **entry transitions** (outside → inside) fire alerts. A person standing inside a zone does not re-trigger — this prevents alert floods.

### Theft Detection — State Machine

Each monitored asset is tracked as `(object_id, last_seen_frame, last_position)`. When an object is absent for more than a threshold number of frames and a person was within proximity of its last known position, a theft alert fires with that person's track ID as the suspect.

---

## Training & Edge Export

### Train the LSTM anomaly classifier

```bash
python scripts/train_anomaly_model.py --epochs 15 --samples 1200
```

Saves the best weights to `models/lstm_anomaly_model.pth`.

### Export to ONNX / TensorRT for edge deployment

```bash
python scripts/export_model.py
```

Outputs `models/lstm_anomaly_model.onnx` and prints TensorRT compilation commands for Jetson.

### Docker on Raspberry Pi

```bash
docker build -f docker/Dockerfile.rpi -t aegisedge-rpi .
docker run --device /dev/video0 -p 5000:5000 aegisedge-rpi
```

---

## Unit Testing

```bash
python -m unittest discover -s tests -v
```

All tests should output `ok` and finish with `OK`. Tests cover zone geometry, tracker ID management, anomaly thresholds, and pipeline hooks.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | venv not activated or install incomplete | Activate venv then re-run `pip install -r requirements.txt` |
| `Cannot open video source: 0` | Webcam in use by another app (Teams, Zoom, etc.) | Close other apps; try `--source 1` for a secondary camera |
| Blank video feed in browser | YOLOv8 downloading weights on first run (~6 MB) | Wait 10–15 seconds for the download to finish |
| `Script execution is disabled on this system` | Windows PowerShell restriction | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator |
| `Port 5000 is already in use` | Another service on port 5000 | Change `dashboard.port` to `5001` in `config/default_config.yaml` |
| FPS extremely low (1–2 FPS) | Heavy model or high resolution on CPU | Use `yolov8n.pt` and keep resolution at `640x480` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Skills Demonstrated

`Computer Vision` · `Edge AI` · `YOLOv8` · `PyTorch` · `LSTM` · `Multi-Object Tracking` · `ByteTrack` · `Optical Flow` · `MobileNetV3` · `ONNX` · `TensorRT` · `OpenCV` · `Flask` · `WebSockets` · `Multithreading` · `Docker` · `Python`
