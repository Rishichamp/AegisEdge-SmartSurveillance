"""
Web Dashboard Server (Flask + WebSockets)
=========================================

This module runs a local web server that provides a real-time monitoring interface.

WHY WEB DASHBOARD & WEBSOCKETS?
===============================
OpenCV's standard output `cv2.imshow()` only works on the local desktop,
and cannot be accessed remotely over the network.

To make the system production-grade, we build a Web Dashboard:
1. **Flask:** Serves the static HTML/CSS/JS frontend files.
2. **MJPEG Stream:** Encodes video frames as JPEG and streams them over HTTP.
3. **WebSockets (Flask-SocketIO):** Enables the backend to immediately "push" alert events
   and hardware metrics (FPS, memory usage) to the browser without the browser reloading.
"""

import os
import time
import queue
import threading
from flask import Flask, render_template, Response, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from typing import Optional, Dict, Any


class DashboardServer:
    """
    Flask + SocketIO server for real-time surveillance dashboard.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, stream_quality: int = 70):
        self.host = host
        self.port = port
        self.stream_quality = stream_quality
        
        # Paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dashboard_dir = os.path.join(self.project_root, "dashboard")
        
        # Initialize Flask & SocketIO
        self.app = Flask(__name__, static_folder=None)
        # cors_allowed_origins="*" allows connection from other devices on local network
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Threading buffers
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self._server_thread = None
        
        self._register_routes()
        self._register_socket_events()
        
    def _register_routes(self):
        """Define Flask HTTP endpoints."""
        
        @self.app.route('/')
        def index():
            """Serve the dashboard main page."""
            return send_from_directory(self.dashboard_dir, "index.html")
            
        @self.app.route('/<path:path>')
        def serve_static(path):
            """Serve static assets (style.css, app.js)."""
            return send_from_directory(self.dashboard_dir, path)
            
        @self.app.route('/logs/alert_screenshots/<filename>')
        def serve_screenshot(filename):
            """Serve alert screenshots saved by the EventLogger."""
            screenshot_dir = os.path.join(self.project_root, "logs", "alert_screenshots")
            return send_from_directory(screenshot_dir, filename)
            
        @self.app.route('/video_feed')
        def video_feed():
            """MJPEG video streaming feed."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
            
        @self.app.route('/api/status')
        def get_status():
            """API endpoint to get system status."""
            return jsonify({
                'status': 'active' if self.running else 'inactive',
                'timestamp': time.time()
            })
            
        @self.app.route('/api/trigger_test', methods=['POST'])
        def trigger_test():
            """API endpoint to trigger a simulated alert."""
            test_alert = {
                'id': f"test-{int(time.time())}",
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                'timestamp_unix': time.time(),
                'type': "Intrusion Detection (Test)",
                'severity': "critical",
                'confidence': 0.95,
                'description': "Simulated restricted zone intrusion test alert from Dashboard Server.",
                'screenshot_url': "",
                'metadata': {'test': True}
            }
            self.send_alert(test_alert)
            return jsonify({'status': 'success', 'alert': test_alert})
            
    def _register_socket_events(self):
        """Define WebSocket events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print("[DASHBOARD] Client connected to WebSocket.")
            emit('system_status', {'connected': True, 'timestamp': time.time()})
            
        @self.socketio.on('ping')
        def handle_ping():
            emit('pong', {'timestamp': time.time()})
            
    def _generate_frames(self):
        """Generator function that yields JPEG compressed video frames."""
        while self.running:
            try:
                # Grab a frame from queue. Blocks up to 1 second.
                frame = self.frame_queue.get(timeout=1.0)
                
                # Compress the frame to JPEG
                # encode_param specifies quality (0-100). Lower = less bandwidth.
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.stream_quality]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except queue.Empty:
                # If no frame arrived, yield a small blank frame or pause briefly
                time.sleep(0.03)
            except Exception as e:
                print(f"[DASHBOARD] Stream error: {e}")
                break
                
    def update_frame(self, frame: np.ndarray):
        """
        Put a new annotated frame into the streaming buffer.
        """
        if not self.running:
            return
            
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
            
    def send_alert(self, alert_event: Dict[str, Any]):
        """
        Push an alert event to the dashboard via WebSockets.
        """
        if not self.running:
            return
            
        # SocketIO runs in a thread-safe context
        self.socketio.emit('alert_event', alert_event)
        
    def send_metrics(self, fps: float, tracks: int, cpu_percent: float, camera_source: str = "Camera 0"):
        """
        Push performance metrics to the dashboard.
        """
        if not self.running:
            return
            
        self.socketio.emit('system_metrics', {
            'fps': round(fps, 1),
            'tracks_active': tracks,
            'cpu_utilization': round(cpu_percent, 1),
            'camera_source': camera_source,
            'timestamp': time.time()
        })
        
    def start(self):
        """Start the Flask server in a background thread."""
        self.running = True
        self._server_thread = threading.Thread(
            target=lambda: self.socketio.run(self.app, host=self.host, port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        self._server_thread.start()
        print(f"[DASHBOARD] Web Server started at http://{self.host}:{self.port}/")
        
    def stop(self):
        """Stop the server."""
        self.running = False
        print("[DASHBOARD] Stopped.")
        # Flask-SocketIO runs as a daemon thread, so stopping the main process closes it.
