"""
Setup script for the Smart Surveillance System package.

This file makes our project installable as a Python package using:
    pip install -e .

The '-e' flag means 'editable' mode - changes to the source code
take effect immediately without reinstalling.

WHY setup.py?
- Makes imports cleaner: `from src.detection import detector` works from anywhere
- Allows us to define console scripts (CLI commands)
- Standard Python packaging practice expected in professional projects
"""

from setuptools import setup, find_packages

setup(
    name="smart-surveillance-system",
    version="1.0.0",
    description="Real-time smart surveillance with AI-powered anomaly detection",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/smart-surveillance-system",
    packages=find_packages(),  # Automatically finds all packages (dirs with __init__.py)
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
        "Flask>=3.0.0",
        "flask-socketio>=5.3.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            # After installation, these commands become available in terminal:
            # $ surveillance-run   -> runs the main surveillance system
            # $ surveillance-demo  -> runs the demo with sample video
            "surveillance-run=scripts.run_surveillance:main",
            "surveillance-demo=scripts.run_demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
