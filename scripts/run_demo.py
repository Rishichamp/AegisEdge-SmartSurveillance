#!/usr/bin/env python3
"""
AegisEdge Surveillance System - Out-of-the-Box Demo
===================================================

This script runs a simulated, high-fidelity demonstration of the Smart
Surveillance system. It does not require a physical webcam or any downloaded
surveillance datasets.

HOW IT WORKS:
=============
1. It initializes the standard `SurveillancePipeline`.
2. It monkey-patches the video stream reader to serve a dynamically generated
   "virtual room" frame (drawing security grids, walls, desks, etc.).
3. It monkey-patches the object detector to return simulated coordinates of moving
   actors (represented as Person #1, Person #2, etc.) performing actions.
4. It simulates the three major alert behaviors:
   - **Zone Intrusion:** Person #2 enters the red polygonal Restricted Area.
   - **Fight/Violence:** Person #3 and Person #4 collide and engage in rapid movement.
   - **Retail/Office Theft:** Person #5 approaches a laptop on a desk, the laptop
     disappears, and Person #5 walks away.
5. It runs the Flask server so you can open http://localhost:5000 and watch the
   dashboard update live!

RUN THE DEMO:
=============
$ python scripts/run_demo.py
"""

import os
import sys
import time
import numpy as np
import cv2
import threading

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline import SurveillancePipeline
from src.detection.detector import Detection
from src.config import get_config


def main():
    print("==================================================================")
    print("              AEGISEDGE MOCK SURVEILLANCE DEMO                    ")
    print("==================================================================")
    print("Initializing demo pipeline with default configuration...")
    
    # Initialize standard pipeline
    pipeline = SurveillancePipeline(config_path="config/default_config.yaml")
    
    # Override settings for demo
    pipeline.config.video.source = "Virtual Security Camera Feed"
    pipeline.config.alerting.cooldown_seconds = 10  # Shorter cooldown for quick demonstration
    pipeline.alert_engine.cooldown_seconds = 10
    
    # Define timeline variables
    frame_counter = 0
    total_demo_frames = 900  # 30 seconds at 30 FPS
    
    # Pre-calculate paths for actors in the virtual room
    # Restricted zone in default_config: [[100, 100], [400, 100], [400, 400], [100, 400]]
    
    # 1. Virtual Feed Generator
    def generate_virtual_frame(idx):
        # Create a sleek dark security grid background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw background security room guides (lines/grid)
        frame[:, :] = (30, 24, 20)  # Very dark BGR blue/gray
        
        # Draw grid lines
        for x in range(0, 640, 80):
            cv2.line(frame, (x, 0), (x, 480), (45, 38, 32), 1)
        for y in range(0, 480, 80):
            cv2.line(frame, (0, y), (640, y), (45, 38, 32), 1)
            
        # Draw room borders / walls (schematic layout)
        cv2.line(frame, (50, 50), (590, 50), (100, 100, 100), 2)
        cv2.line(frame, (50, 430), (590, 430), (100, 100, 100), 2)
        cv2.line(frame, (50, 50), (50, 430), (100, 100, 100), 2)
        cv2.line(frame, (590, 50), (590, 430), (100, 100, 100), 2)
        
        # Draw static desk in the room
        cv2.rectangle(frame, (260, 260), (340, 320), (80, 60, 50), -1)  # Table top
        cv2.rectangle(frame, (260, 260), (340, 320), (120, 100, 90), 2)  # Table border
        cv2.putText(
            frame, 
            "DESK", 
            (285, 295), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (200, 200, 200), 
            1, 
            cv2.LINE_AA
        )
        
        # Render a text watermark at the bottom of the feed
        cv2.putText(
            frame, 
            "VIRTUAL FEED - LAB SIMULATION", 
            (30, 460), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (150, 150, 150), 
            1, 
            cv2.LINE_AA
        )
        
        # Overlay system time
        sys_time = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(
            frame, 
            f"CAM 01  {sys_time}", 
            (480, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        # Draw visual markers for actors
        detections = get_virtual_detections(idx)
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            
            if det.class_name == "person":
                # Draw a small indicator circle representing the person's position
                cv2.circle(frame, center, 8, (255, 191, 0), -1)
                # Draw small heading direction vector line
                cv2.line(frame, center, (center[0] + 10, center[1] + 10), (255, 255, 255), 2)
            elif det.class_name == "laptop":
                # Draw laptop symbol on the desk
                cv2.rectangle(frame, (x1+4, y1+4), (x2-4, y2-4), (0, 165, 255), -1)
                
        return frame

    # 2. Virtual Detections Generator
    def get_virtual_detections(idx):
        # Scale down loop index if it goes past maximum
        local_idx = idx % total_demo_frames
        detections = []
        
        # A. Restricted Zone Intruder: Person #2 (Frames 60 - 200)
        if 60 <= local_idx <= 200:
            # Person starts at coordinates (80, 80) and walks deep into the restricted zone
            prog = (local_idx - 60) / 140.0
            x = 80 + prog * 180.0  # Moves 80 -> 260
            y = 80 + prog * 180.0  # Moves 80 -> 260
            
            detections.append(Detection(
                bbox=np.array([x-20, y-50, x+20, y+50]),
                confidence=0.88,
                class_id=0,
                class_name="person",
                track_id=2
            ))
            
        # B. Fight Scenario: Person #3 and Person #4 (Frames 240 - 450)
        if 240 <= local_idx <= 450:
            # Person #3 starts at (520, 120) and walks left
            # Person #4 starts at (380, 120) and walks right
            # They meet at (450, 120) around frame 300, and show chaotic motion
            
            if local_idx < 300:
                prog = (local_idx - 240) / 60.0
                p3_x = 520 - prog * 70.0  # Moves 520 -> 450
                p4_x = 380 + prog * 70.0  # Moves 380 -> 450
                p3_y = 120
                p4_y = 120
            elif local_idx <= 380:
                # Altercation phase: rapid back-and-forth movement (oscillation)
                osc = 15.0 * np.sin(local_idx * 0.8)
                p3_x = 450 + osc
                p4_x = 440 - osc
                p3_y = 120 + 8.0 * np.cos(local_idx * 0.5)
                p4_y = 120 - 8.0 * np.cos(local_idx * 0.5)
            else:
                # Separation phase: they walk away in opposite directions
                prog = (local_idx - 380) / 70.0
                p3_x = 450 + prog * 100.0
                p4_x = 430 - prog * 100.0
                p3_y = 120
                p4_y = 120
                
            detections.append(Detection(
                bbox=np.array([p3_x-20, p3_y-50, p3_x+20, p3_y+50]),
                confidence=0.91,
                class_id=0,
                class_name="person",
                track_id=3
            ))
            detections.append(Detection(
                bbox=np.array([p4_x-20, p4_y-50, p4_x+20, p4_y+50]),
                confidence=0.87,
                class_id=0,
                class_name="person",
                track_id=4
            ))
            
        # C. Theft Scenario: Laptop and Person #5 (Frames 500 - 800)
        # Stationary laptop on the desk
        # Person #5 walks near desk, picks up laptop, laptop disappears, walks away
        
        # Laptop stays on desk (300, 290) from frame 0 until picked up at frame 620
        laptop_present = local_idx < 620
        if laptop_present:
            detections.append(Detection(
                bbox=np.array([290, 280, 310, 300]),
                confidence=0.95,
                class_id=63,
                class_name="laptop",
                track_id=10  # Tracked object ID
            ))
            
        if 500 <= local_idx <= 800:
            # Person #5 starts at (480, 350) and walks toward desk (300, 290)
            if local_idx < 620:
                prog = (local_idx - 500) / 120.0
                p5_x = 480 - prog * 180.0  # Moves 480 -> 300
                p5_y = 350 - prog * 60.0   # Moves 350 -> 290
            else:
                prog = (local_idx - 620) / 180.0
                p5_x = 300 + prog * 200.0  # Walks away to the right: 300 -> 500
                p5_y = 290 + prog * 60.0   # 290 -> 350
                
            detections.append(Detection(
                bbox=np.array([p5_x-20, p5_y-50, p5_x+20, p5_y+50]),
                confidence=0.90,
                class_id=0,
                class_name="person",
                track_id=5
            ))
            
        return detections

    # Patch the video stream read method to return our virtual frames
    def custom_read():
        nonlocal frame_counter
        frame_counter += 1
        frame = generate_virtual_frame(frame_counter)
        return frame
        
    pipeline.video_stream.read = custom_read
    pipeline.video_stream.start = lambda: None  # Dummy to prevent starting webcam thread
    pipeline.video_stream.stop = lambda: None
    
    # Patch the detector to return our virtual detections
    def custom_detect_and_track(frame, tracker="bytetrack.yaml", persist=True):
        return get_virtual_detections(frame_counter)
        
    pipeline.detector.detect_and_track = custom_detect_and_track
    
    # Patch the fight detector's motion calculator to return high energy during the fight window
    def custom_analyze(frame, person_detections, lstm_predictions):
        local_idx = frame_counter % total_demo_frames
        if 290 <= local_idx <= 350:
            # Force fight flag to True during meeting frame window
            return True, 0.82, [3, 4]
        return False, 0.0, []
        
    pipeline.fight_detector.analyze = custom_analyze
    
    # We will simulate feature extraction output
    def custom_extract_from_frame(frame, bbox):
        # Return a simple mock feature vector matching MobileNetV3 dim (576)
        # Shift slightly based on frame index to simulate posture variance
        local_idx = frame_counter % total_demo_frames
        base = np.zeros(576)
        
        # Inject custom variance if actors 3 & 4 are fighting
        if 290 <= local_idx <= 350:
            # High noise to trigger the LSTM fallback variance classifier
            return base + np.random.normal(0, 0.35, 576)
        return base + np.random.normal(0, 0.05, 576)
        
    pipeline.feature_extractor.extract_from_frame = custom_extract_from_frame

    # Start the patched pipeline
    print("Starting Flask dashboard and simulation threads...")
    pipeline.start()
    
    print("\n" + "="*60)
    print("      DEMO IS RUNNING SUCCESFULLY!")
    print("      Open your web browser and navigate to:")
    print("      --> http://localhost:5000/ <--")
    print("="*60 + "\n")
    print("Press Ctrl+C in this terminal to stop the demo.")
    
    # Log progress status in console for transparency
    try:
        while pipeline.running:
            local_idx = frame_counter % total_demo_frames
            if local_idx == 0:
                print("[DEMO] Loop restart. Simulating normal scene...")
            elif local_idx == 60:
                print("[DEMO] Actor #2 moving toward restricted zone...")
            elif local_idx == 120:
                print("[DEMO] Restricted Zone Intrusion Detected! Logging and saving screenshot...")
            elif local_idx == 240:
                print("[DEMO] Actor #3 and Actor #4 entering frame...")
            elif local_idx == 295:
                print("[DEMO] Altercation Detected! Fusing motion & position. Logging threat...")
            elif local_idx == 500:
                print("[DEMO] Actor #5 approaching laptop stationary on desk...")
            elif local_idx == 620:
                print("[DEMO] Laptop removed from desk by Actor #5. Disappearance Theft Alert Triggered!")
                
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[DEMO] Stopping simulation...")
    finally:
        pipeline.stop()
        print("[DEMO] Shutdown complete.")


if __name__ == "__main__":
    main()
