# Edge Deployment & Containerization Guide
==========================================

This document details how to deploy the AegisEdge Smart Surveillance system on hardware edge boards (NVIDIA Jetson Nano/Orin or Raspberry Pi 4/5) using Docker containerization and model compiler tools.

---

## 1. NVIDIA Jetson Nano / Orin Deployment

NVIDIA Jetson devices feature CUDA Cores and Tensor Cores designed for low-latency deep learning inference. To compile the system for Jetson, follow these steps.

### Step 1.1. Set up NVIDIA Container Toolkit
Ensure your Jetson board has JetPack installed. The NVIDIA Container Toolkit allows Docker containers to access the underlying Jetson GPU.
```bash
# Verify nvidia runtime is active
docker info | grep -i nvidia
```

### Step 1.2. Build the Jetson Docker Container
We use our custom CUDA-enabled Dockerfile:
```bash
docker build -t aegisedge-jetson -f docker/Dockerfile.jetson .
```

### Step 1.3. Run the Container on the Edge Board
To run the container, you must pass the `--runtime nvidia` flag to expose the GPU, and pass the video devices (`/dev/video0`) to let the container access physical USB webcams:
```bash
docker run --runtime nvidia \
  --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  aegisedge-jetson
```
* **Flags Breakdown:**
  * `--runtime nvidia`: Enables CUDA acceleration inside the container.
  * `--device /dev/video0:/dev/video0`: Mounts USB camera 0 into the container.
  * `-p 5000:5000`: Maps the Flask Dashboard to http://localhost:5000.
  * `-v`: Maps local log folder to retain screenshots and events outside the container.

---

## 2. Raspberry Pi 4/5 Deployment

Since Raspberry Pi lacks an NVIDIA GPU, the system runs in **CPU-Only mode**. We optimize performance by stripping out heavy CUDA packages.

### Step 2.1. Build the Pi CPU Container
Compile using the lightweight python-slim base:
```bash
docker build -t aegisedge-rpi -f docker/Dockerfile.rpi .
```

### Step 2.2. Run the Container on the Pi
```bash
docker run --device /dev/video0:/dev/video0 \
  -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  aegisedge-rpi
```

### Step 2.3. Optimization Tips for Raspberry Pi
* **Frame Skip:** Set `performance.frame_skip: 3` in `config/rpi_config.yaml` to process every 3rd frame (10 FPS instead of 30), which reduces CPU workload by 66%.
* **YOLO Model:** Always use `yolov8n.pt` (YOLOv8 Nano), which contains only 3.2 Million parameters.
* **Resolution:** Cap video resolution to `640x480`.

---

## 3. Production Dashboard Server (Gunicorn & Nginx)

Flask's built-in WSGI server (`app.run()`) is single-threaded and not designed for production scaling. If multiple security guards access the dashboard, it will block.

To deploy the dashboard in a production environment:

### Step 3.1. Deploy Gunicorn
Use **Gunicorn** (Green Unicorn) as the WSGI server inside the Docker container:
```bash
# Run with 4 worker threads supporting Socket.io WebSockets
gunicorn --worker-class eventlet -w 1 --threads 4 -b 0.0.0.0:5000 app:app
```

### Step 3.2. Nginx Reverse Proxy
Deploy **Nginx** as a frontend proxy to handle secure SSL connections, cache static assets (`style.css`, `app.js`), and route WebSocket connections:
```nginx
server {
    listen 80;
    server_name surveillance.local;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Handle WebSocket connections specifically
    location /socket.io {
        proxy_pass http://localhost:5000/socket.io;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```
