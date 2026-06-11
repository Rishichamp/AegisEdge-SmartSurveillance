# System Setup Guide — AegisEdge Smart Surveillance
===================================================

This document provides step-by-step instructions to set up the AegisEdge Smart Surveillance environment on a desktop machine, laptop, or edge computer (Raspberry Pi/Jetson).

---

## 1. Prerequisites

Ensure you have the following installed on your host system:
* **Python 3.8 to 3.11** (PyTorch and YOLOv8 are highly stable in this range).
* **pip** (Python package installer).
* **Git** (for cloning).
* **C++ Build Tools** (Required for compiling certain dependencies like `scipy` or `psutil` if pre-compiled wheels aren't available).
  - *Windows:* Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++".
  - *Ubuntu/Debian:* Run `sudo apt install build-essential python3-dev`

---

## 2. Local Desktop Installation (Windows/macOS/Linux)

Follow these terminal commands to initialize the workspace:

### Step 2.1. Clone the Codebase
```bash
git clone https://github.com/yourusername/smart-surveillance-system.git
cd smart-surveillance-system
```

### Step 2.2. Create a Virtual Environment
Virtual environments prevent library dependency conflicts between projects.
```bash
# Windows (PowerShell or Command Prompt)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 2.3. Install Dependencies
Run pip install to fetch deep learning libraries and utilities:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
*(Note: If you do not have an NVIDIA GPU, this installs the standard PyTorch CPU version, which is perfectly suitable for running our local dashboard demo).*

### Step 2.4. Verify Setup
Run our test suite to verify that imports and structures function correctly:
```bash
python -m unittest discover -s tests
```

---

## 3. Running the System

AegisEdge provides two entry scripts for execution:

### Option A: Out-of-the-Box Demo (Recommended)
This runs the system using a virtual room generator and mock actor paths, allowing you to test the web dashboard and alert logs immediately without any webcams or video files.
```bash
python scripts/run_demo.py
```
* **Step 1:** Run the script in your terminal.
* **Step 2:** Open your web browser and navigate to **`http://localhost:5000`**.
* **Step 3:** Watch the live visual grid, track lines, and incoming alert events! Click on alerts to open the Investigation Modal.

### Option B: Live Surveillance Stream
To run the system on a live webcam (device index 0) or an RTSP network camera:
```bash
# Run on default webcam
python scripts/run_surveillance.py

# Run on a local video file
python scripts/run_surveillance.py --source path/to/your/video.mp4

# Run with a specific configuration file (e.g. Raspberry Pi settings)
python scripts/run_surveillance.py --config config/rpi_config.yaml
```

---

## 4. Troubleshooting Configuration Errors

### 4.1. OpenCV "Cannot open video source"
* **Reason:** Your webcam index might be different, or another program (like Zoom, Teams) is using it.
* **Fix:** Open `config/default_config.yaml` and change `video.source` from `0` to `1` or `2`. Ensure other camera apps are closed.

### 4.2. PyTorch "CUDA out of memory" (If running on GPU)
* **Reason:** Your GPU memory is filled by other background processes.
* **Fix:** Decrease resolution to `640x480` and use the smaller YOLO model (`yolov8n.pt`) in `config/default_config.yaml`.
