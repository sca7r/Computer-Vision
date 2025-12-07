# Hand and Face Tracking with Sleep Detection

A real-time computer vision system that tracks hands and facial features while detecting sleep states using Eye Aspect Ratio (EAR) analysis. Built with Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-orange.svg)


## Features

-  **Real-time Hand Tracking**: Detects and tracks up to 2 hands with 21 landmark points per hand
-  **Facial Landmark Detection**: Tracks 478 facial landmarks using MediaPipe Face Mesh
-  **Sleep/Awake Detection**: Monitors eye closure using Eye Aspect Ratio (EAR) algorithm
-  **Live Statistics**: Displays real-time EAR values, status, and blink counter
-  **Webcam Support**: Works with standard webcams and supports multiple camera indices

## Demo

The system provides:
- Green dots and connecting lines for hand landmarks
- Face mesh contours showing key facial features
- Real-time EAR (Eye Aspect Ratio) measurement
- Status indicator: AWAKE (green) or SLEEPING (red)


## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- Linux/Windows/macOS

### Setup

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```
## Run the script 

```bash
python main.py
```
