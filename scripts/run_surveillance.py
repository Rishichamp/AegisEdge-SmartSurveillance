#!/usr/bin/env python3
"""
AegisEdge Surveillance System - Main Command Line Entry Point
============================================================

This script allows you to launch the Smart Surveillance Edge AI system. It supports
overriding the configuration parameters (like video source, processing device)
directly from the terminal.

HOW TO RUN:
===========
1. Run with default settings (Webcam 0, CPU):
   $ python scripts/run_surveillance.py

2. Run on a video file with CPU:
   $ python scripts/run_surveillance.py --source data/samples/test_video.mp4

3. Run on a specific GPU:
   $ python scripts/run_surveillance.py --source 0 --device cuda:0

4. Run with a custom configuration:
   $ python scripts/run_surveillance.py --config config/jetson_config.yaml
"""

import os
import sys
import argparse
import time

# Ensure the project root is in the python path (so "from src... import" works)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline import SurveillancePipeline
from src.config import get_config


def main():
    parser = argparse.ArgumentParser(
        description="AegisEdge Smart Surveillance System - Edge AI Real-Time Monitor"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default_config.yaml",
        help="Path to YAML configuration file (default: config/default_config.yaml)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=None,
        help="Video source override (integer for webcam, string for video file or RTSP stream)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Inference execution device override ('cpu', 'cuda', '0', etc.)"
    )
    
    args = parser.parse_args()
    
    print("==================================================================")
    print("                AEGISEDGE SMART SURVEILLANCE AI                   ")
    print("==================================================================")
    print(f"Loading system configuration: {args.config}")
    
    # Check if the config file actually exists
    if not os.path.exists(args.config):
        # Check relative to project root
        alternative_path = os.path.join(project_root, args.config)
        if os.path.exists(alternative_path):
            args.config = alternative_path
        else:
            print(f"Error: Configuration file not found at '{args.config}'")
            sys.exit(1)
            
    # Initialize pipeline
    try:
        pipeline = SurveillancePipeline(config_path=args.config)
        
        # Apply command-line overrides to config variables if provided
        if args.source is not None:
            # Check if source is a number string (webcam)
            if args.source.isdigit():
                pipeline.config.video.source = int(args.source)
            else:
                pipeline.config.video.source = args.source
            # Recreate video stream capture with the overridden source
            pipeline.video_stream.source = pipeline.config.video.source
            print(f"[OVERRIDE] Video source set to: {pipeline.config.video.source}")
            
        if args.device is not None:
            pipeline.config.detection.device = args.device
            pipeline.detector.device = args.device
            # Re-map features devices
            import torch
            pipeline.feature_extractor.device = torch.device(args.device)
            pipeline.feature_extractor.features.to(pipeline.feature_extractor.device)
            pipeline.temporal_predictor.device = torch.device(args.device)
            pipeline.temporal_predictor.model.to(pipeline.temporal_predictor.device)
            print(f"[OVERRIDE] Execution device set to: {pipeline.config.detection.device}")

        # Start the pipeline
        pipeline.start()
        
        # Keep the main process alive, waiting for shutdown trigger
        while pipeline.running:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt detected. Initiating clean exit...")
    except Exception as e:
        print(f"\n[CRITICAL] System failure: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop and release resources
        if 'pipeline' in locals():
            pipeline.stop()
        print("[SYSTEM] AegisEdge Surveillance System stopped cleanly. Goodbye.")


if __name__ == "__main__":
    main()
