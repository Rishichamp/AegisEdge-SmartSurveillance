"""
Video Stream Handler
====================

This module provides a thread-safe video capture abstraction that works
with USB cameras, RTSP network cameras, and video files.

WHY THREAD-SAFE VIDEO CAPTURE?
================================
Imagine you're watching a live camera feed. The camera produces 30 frames
per second. If our AI detection takes 100ms per frame (10 FPS), we can only
process 10 of those 30 frames.

Problem WITHOUT threading:
    1. Read frame (takes ~33ms waiting for camera)
    2. Run detection (takes ~100ms)
    3. Read next frame (but we missed 3 frames while detecting!)
    → Result: Laggy, choppy video with increasing delay

Solution WITH threading:
    Thread 1 (Capture): Continuously reads frames into a buffer
    Thread 2 (Main):    Grabs the latest frame from buffer whenever ready
    → Result: Always have the most recent frame, no lag
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Union
from queue import Queue


class VideoStream:
    """
    Thread-safe video capture from cameras or video files.
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self._capture = None
        self._thread = None
        self._running = False
        self._frame_queue = Queue(maxsize=buffer_size)
        self._lock = threading.Lock()
        
        self._frame_count = 0
        self._fps_actual = 0.0
        self._last_fps_time = 0.0
        self._fps_frame_count = 0
        self._is_file = False
        self._total_frames = 0
    
    def start(self) -> 'VideoStream':
        """Open the video source and start the capture thread."""
        # Check if source is integer (e.g. webcam ID) or string (file/RTSP)
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                source_val = int(self.source)
            else:
                source_val = self.source
        except ValueError:
            source_val = self.source

        self._capture = cv2.VideoCapture(source_val)
        
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source}\n"
                f"If using a webcam, ensure it is connected and not in use."
            )
        
        # Set dimensions
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._is_file = self._total_frames > 0
        
        # Get actual resolution
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        
        print(f"[VIDEO] Opened source: {self.source}")
        print(f"[VIDEO] Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        self._running = True
        self._last_fps_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return self
    
    def _capture_loop(self):
        while self._running:
            ret, frame = self._capture.read()
            
            if not ret:
                if self._is_file:
                    # Loop video files back to the beginning
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue
            
            # Optional resize
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Drop older frames if the queue is full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Exception:
                    pass
            
            self._frame_queue.put(frame)
            
            self._frame_count += 1
            self._fps_frame_count += 1
            
            current_time = time.time()
            elapsed = current_time - self._last_fps_time
            if elapsed >= 1.0:
                self._fps_actual = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = current_time
    
    def read(self) -> Optional[np.ndarray]:
        """Get the most recent frame from the buffer."""
        if not self._running:
            return None
        
        try:
            # Non-blocking fetch
            return self._frame_queue.get(timeout=0.05)
        except Exception:
            return None
    
    def stop(self):
        """Stop the capture thread and release video capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        print(f"[VIDEO] Stopped. Total frames: {self._frame_count}")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def actual_fps(self) -> float:
        return self._fps_actual
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def is_file_source(self) -> bool:
        return self._is_file
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
