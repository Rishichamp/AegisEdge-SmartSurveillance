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

import hmac
import os
import time
import threading
from flask import Flask, Response, send_from_directory, jsonify, request
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
        
        # Optional login. The dashboard shows the live camera feed and binds to
        # all interfaces by default, so anyone on the network could watch it.
        # Set AEGIS_DASHBOARD_PASSWORD (and optionally AEGIS_DASHBOARD_USER,
        # default "admin") to require HTTP Basic auth for every page, the
        # video feed and the WebSocket.
        self.auth_user = os.getenv("AEGIS_DASHBOARD_USER", "admin")
        self.auth_password = os.getenv("AEGIS_DASHBOARD_PASSWORD", "")
        if not self.auth_password and host not in ("127.0.0.1", "localhost"):
            print(f"[DASHBOARD] WARNING: serving the camera feed on {host} without a password. "
                  "Set AEGIS_DASHBOARD_PASSWORD to require a login.")

        # Initialize Flask & SocketIO
        self.app = Flask(__name__, static_folder=None)
        # cors_allowed_origins="*" allows connection from other devices on local network
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Latest frame, shared by every viewer. (A size-1 queue meant each
        # frame went to just one viewer: two open tabs got half the FPS each.)
        self._frame = None
        self._frame_id = 0
        self._frame_cond = threading.Condition()
        self.running = False
        self._server_thread = None
        
        self._register_routes()
        self._register_socket_events()
        
    def _authorized(self) -> bool:
        if not self.auth_password:
            return True
        auth = request.authorization
        return bool(auth) and hmac.compare_digest(auth.username or "", self.auth_user) \
            and hmac.compare_digest(auth.password or "", self.auth_password)

    def _register_routes(self):
        """Define Flask HTTP endpoints."""

        @self.app.before_request
        def require_login():
            if not self._authorized():
                return Response("Login required", 401,
                                {"WWW-Authenticate": 'Basic realm="AegisEdge dashboard"'})
        
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
            if not self._authorized():
                return False  # reject the WebSocket
            print("[DASHBOARD] Client connected to WebSocket.")
            emit('system_status', {'connected': True, 'timestamp': time.time()})
            
        @self.socketio.on('ping')
        def handle_ping():
            emit('pong', {'timestamp': time.time()})
            
    def _generate_frames(self):
        """Generator function that yields JPEG compressed video frames."""
        last_id = 0  # frame ids start at 1; 0 = nothing published yet
        while self.running:
            try:
                # Wait (up to 1 s) for a frame newer than the last one sent.
                with self._frame_cond:
                    self._frame_cond.wait_for(lambda: self._frame_id != last_id or not self.running,
                                              timeout=1.0)
                    if self._frame_id == last_id:
                        continue
                    frame, last_id = self._frame, self._frame_id
                
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
                       
            except Exception as e:
                print(f"[DASHBOARD] Stream error: {e}")
                break
                
    def update_frame(self, frame: np.ndarray):
        """
        Put a new annotated frame into the streaming buffer.
        """
        if not self.running:
            return

        with self._frame_cond:
            self._frame = frame
            self._frame_id += 1
            self._frame_cond.notify_all()
            
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
        with self._frame_cond:
            self._frame_cond.notify_all()
        print("[DASHBOARD] Stopped.")
        # Flask-SocketIO runs as a daemon thread, so stopping the main process closes it.
