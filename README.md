# AegisEdge: Smart Surveillance System (Edge AI)
=================================================

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-brightgreen.svg)](https://github.com/ultralytics/ultralytics)

AegisEdge is a real-time, high-performance Smart Surveillance Edge AI system designed to run on local workstations and edge platforms (NVIDIA Jetson Nano/Orin, Raspberry Pi 4/5). It processes live video streams (USB cameras, RTSP network feeds, or local files) to track multiple objects and detect security anomalies.

> **Placement Portfolio Highlight:** This system is built from scratch with clean modular separation, a thread-safe double-buffered capture engine, a hybrid AI/rule-based analytics pipeline, and a production-grade Web Dashboard utilizing WebSockets.

---

## 🚀 Key Features

* **Multi-Object Tracking:** Utilizes YOLOv8 and ByteTrack to track people and asset bounding boxes across frames with Kalman filters to reduce track ID switches.
* **Restricted Zone Intrusions:** Supports custom polygonal zone definitions. Computes feet coordinates and performs point-in-polygon ray-casting tests to trigger warnings on entry/exit transitions.
* **Retail / Office Asset Theft:** Uses a spatial-temporal state machine to detect when monitored assets (laptops, purses, suitcases) disappear from a scene while an associated person is nearby.
* **Fight & Violence Detection:** Combines spatial proximity check rules, Farneback Dense Optical Flow magnitude averages, and a PyTorch LSTM posture sequence classifier.
* **Premium Glassmorphic Web Dashboard:** Real-time monitoring UI (Flask + WebSockets Socket.io) serving low-latency MJPEG video streams, hardware metrics (inference FPS, CPU utilization), and dynamic threat alert cards with modal investigation windows.
* **Throttled Alert Engine:** Built-in alert cooldown system preventing notification floods during continuous threats. Log entries are written to append-only JSONL files alongside alert screenshots.

---

## 🛠️ Technology Stack

* **Core & Processing:** Python, NumPy, SciPy (Spatial algebra)
* **Computer Vision:** OpenCV (Video capture, Farneback Optical Flow, image annotation)
* **Deep Learning:** PyTorch (LSTM action classifier), Torchvision (MobileNetV3 spatial posture feature extractor), Ultralytics YOLOv8 (Detector + tracker)
* **Dashboard & Networking:** Flask, Flask-SocketIO (WebSocket events), HTML5, CSS3, JavaScript (ES6, Socket.io client)
* **Edge Compilation:** ONNX, NVIDIA TensorRT (`trtexec` quantization optimization)
* **Containerization:** Docker (ARM64 Dockerfiles for Jetson boards and Raspberry Pi)

---

## 📂 Project Structure

```directory
smart-surveillance-system/
├── config/                  # YAML settings files for different run environments
│   ├── default_config.yaml  # Default local desktop settings
│   ├── jetson_config.yaml   # Optimized for NVIDIA Jetson CUDA/TensorRT
│   └── rpi_config.yaml      # CPU-optimized frame-skipping settings for Raspberry Pi
├── dashboard/               # Web Dashboard frontend assets
│   ├── index.html           # Structure markup (Outfit & Plus Jakarta Sans typography)
│   ├── style.css            # Glassmorphism dark-mode styles
│   └── app.js               # Socket.io WebSocket rendering client logic
├── docker/                  # Docker container recipes
│   ├── Dockerfile.jetson    # NVIDIA L4T CUDA pre-installed container configuration
│   └── Dockerfile.rpi       # CPU-only slim python container configuration
├── docs/                    # Deep-dive educational documents
│   ├── ARCHITECTURE.md      # Data-flow diagrams, thread architecture, design patterns
│   ├── COMPONENTS.md        # Class-by-class code explanations and configurations
│   ├── SETUP_GUIDE.md       # Step-by-step local workstation environment setups
│   ├── HOW_IT_WORKS.md      # Mathematical equations, PIP ray-casting, and optical flow
│   ├── EDGE_DEPLOYMENT.md   # Jetpack Docker commands and USB device bindings
│   ├── TRAINING_GUIDE.md    # Custom dataset preparation and LSTM train scripts
│   └── INTERVIEW_PREP.md    # Portfolio elevator pitches and recruiter QA cheat sheet
├── models/                  # PyTorch, ONNX, and TensorRT engine weight checkpoints
├── scripts/                 # Execution entry points
│   ├── run_surveillance.py  # Main pipeline script with command overrides
│   ├── run_demo.py          # Zero-dependency virtual demo play script
│   ├── train_anomaly_model.py # PyTorch LSTM network training script
│   └── export_model.py      # ONNX export and TensorRT compilation script
├── src/                     # Core source package
│   ├── alerting/            # Cooldown alert engines and JSONL log writers
│   ├── anomaly/             # Threat classifiers (Fight, Zone, Theft) and feature extractors
│   ├── detection/           # YOLOv8 wrappers and coordinate history trackers
│   ├── models/              # PyTorch RNN/LSTM neural network definitions
│   ├── utils/               # Double-buffered video stream queues and math geometry
│   ├── visualization/       # Frame annotators and Flask web servers
│   └── config.py            # Configuration loader and accessor
├── tests/                   # Complete unit test suite (zone tests, tracker tests, anomaly tests)
├── requirements.txt         # Pip package dependencies list
└── setup.py                 # Package setup installer
```

---

## ⏱️ Quick Start — Out-of-the-Box Demo

To run a simulated room demonstration showing all features, follow these steps (no webcam or video downloads required!):

### 1. Configure the Virtual Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/smart-surveillance-system.git
cd smart-surveillance-system

# Initialize Python virtual environment
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate
# Activate venv (Linux/macOS)
source venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Run the Demo Script
```bash
python scripts/run_demo.py
```

### 3. View the Dashboard
Open your browser and navigate to **[http://localhost:5000](http://localhost:5000)**.
* **Intrusion Simulation:** Watch Actor #2 enter the red restricted polygon. An alarm immediately logs in the system history table and pops up in the alerts panel.
* **Altercation Simulation:** Watch Actor #3 and #4 meet and engage in rapid kinetic motion, triggering the Fight Alert.
* **Theft Simulation:** Watch Actor #5 pick up a laptop from a desk. The laptop disappears from the tracking system, triggering a Theft Alert.
* **Investigation:** Click "View" on any alert in the log or sidebar to load the captured alert screenshot and metadata in the details modal!

---

## 📖 Detailed Guides Index

For comprehensive details on how to build, run, optimize, and speak about this project, refer to the documentation files in the `docs/` directory:

1. 🏛️ **[System Architecture](docs/ARCHITECTURE.md)**: Explore the data-flow diagrams, three-tiered thread model, and key design patterns (double-buffering, gate-filtering).
2. 🧩 **[Class Components](docs/COMPONENTS.md)**: Look up function signatures, class methods, and settings for each file in `src/`.
3. ⚙️ **[Setup & Run Guide](docs/SETUP_GUIDE.md)**: Step-by-step setup guides for local laptops, video overrides, and troubleshooting.
4. 🧮 **[How it Works (Mathematics)](docs/HOW_IT_WORKS.md)**: Learn about the ray-casting algorithm, Farneback optical flow calculations, and LSTM cell equations.
5. 🐳 **[Edge Deployment](docs/EDGE_DEPLOYMENT.md)**: Guide to containerizing the application for Jetson Nano/Orin or Raspberry Pi, mounting camera devices, and setting up Nginx proxies.
6. 🧠 **[Model Training Guide](docs/TRAINING_GUIDE.md)**: Preprocess video clips into sequence feature sets and train the LSTM network on custom classes.
7. 💼 **[Interview Preparation](docs/INTERVIEW_PREP.md)**: Study elevator pitches, design decisions, and common recruiter questions to excel in placement interviews.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
