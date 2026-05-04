# Computer Vision & Monitoring System 

> Real-time drowsiness detection with hand & face tracking, audio alarm, email/SMS alerts, and a live web dashboard, built with Python, MediaPipe Tasks API, and FastAPI.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

##  Overview

A production-grade monitoring system that uses computer vision to detect drowsiness in real time. It tracks facial landmarks and hand positions via webcam, calculates the **Eye Aspect Ratio (EAR)** to determine if a person is falling asleep, and fires multi-channel alerts, all while streaming a live feed to a web dashboard.

---

##  Features

| Feature | Description |
|---|---|
|  **Sleep Detection** | Eye Aspect Ratio (EAR) algorithm with configurable threshold and frame window |
|  **Hand Tracking** | 21-landmark hand tracking for up to 2 hands simultaneously |
|  **Multi-Person** | Independent sleep state tracked per face (up to 10 simultaneous) |
|  **Audio Alarm** | Procedurally generated beep alarm with cooldown, runs in background thread |
|  **Email Alerts** | HTML email sent via SMTP (Gmail-ready) when sleeping is detected |
|  **SMS Alerts** | Twilio SMS notification with configurable recipients |
|  **Live Dashboard** | FastAPI + MJPEG stream with real-time stats, per-face table, and alert log |
|  **Config File** | All settings in a single `config.yaml` — no code changes needed |

---

##  Dashboard Preview

The web dashboard runs at `http://localhost:8000` and shows:
- **Live camera feed** via MJPEG stream
- **FPS, face count, sleeping count** stat cards
- **Per-face table** with EAR value, AWAKE/SLEEPING badge, and blink(times asleep) counter
- **Alert log** with timestamps for every sleep, email, and SMS event

---

## How It Works

### Eye Aspect Ratio (EAR)
```
EAR = (|p2−p6| + |p3−p5|) / (2 × |p1−p4|)
```
When EAR drops below `ear_threshold` for `consec_frames` consecutive frames, the person is flagged as **SLEEPING** and all configured alerts fire.

### Architecture
```
Webcam
  └── cv2.VideoCapture
        └── mp.Image
              ├── FaceLandmarker (thread 1) ──► EAR calc ──► PersonState ──► Alerts
              └── HandLandmarker (thread 2) ──► Draw landmarks

SharedState (thread-safe)
  ├── JPEG frame  ──► FastAPI /stream  ──► Browser MJPEG
  ├── Stats       ──► FastAPI /stats   ──► Dashboard cards
  └── Alert log   ──► FastAPI /alerts  ──► Alert log table
```

---

## Demo

This demo shows the system running in a car-like driver monitoring setup. A camera is positioned to observe the driver seat while the application tracks both facial landmarks and hand landmarks in real time.

The sleep/awake detection is based on the Eye Aspect Ratio (EAR) algorithm. When the driver’s eyes remain closed for more than 2 seconds, the system marks the driver as sleeping and triggers an audio alert to get their attention.

The demo highlights:
- Real-time face and hand tracking
- EAR-based drowsiness detection
- Awake vs sleeping status updates
- Sound alert when prolonged eye closure is detected

<video src="./assets/cv.mp4" controls width="720"></video>

[Watch the demo video](https://github.com/sca7r/hComputer-Vision/raw/main/assets/cv.mp4)


---

## Project Structure

```
hand and face tracking with leep awake detection/
│
├── main.py              # Entry point — detection loop
├── config.yaml          # All settings in one place
├── requirements.txt     # Python dependencies
├── download_models.py   # One-time model file downloader
├── assets/              # Demo media for GitHub README
│
├── src/
│   ├── api.py           # FastAPI server + web dashboard
│   ├── alerts.py        # Email and SMS alerters
│   ├── config.py        # YAML config loader with validation
│   └── state.py         # Thread-safe shared state (detection ↔ API)
│
└── models/
    ├── face_landmarker.task
    └── hand_landmarker.task
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sca7r/hand-and-face-tracking-with-sleep-awake-detection.git
cd hand-and-face-tracking-with-sleep-awake-detection
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model files
```bash
python3 download_models.py
```

### 5. Run
```bash
python3 main.py
```

Open `http://localhost:8000` in your browser to see the live dashboard.

Press **`q`** in the OpenCV window to quit.

---

##  Configuration

All settings are in `config.yaml`. Edit it once — no code changes needed.


---

##  Setting Up Email Alerts (Gmail)

1. Enable **2-Step Verification** on your Google account
2. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a new App Password and paste it as `email.password` in `config.yaml`
4. Set `email.enabled: true`

> ⚠️ Never commit your real password to Git. Use environment variables or a `.env` file for production.

---

##  Setting Up SMS Alerts (Twilio)

1. Create a free account at [twilio.com](https://www.twilio.com)
2. Get your **Account SID**, **Auth Token**, and a **Twilio phone number**
3. Fill in the `sms` section in `config.yaml`
4. Set `sms.enabled: true`

---

##  CLI Options

```bash
python3 main.py                        # uses config.yaml (default)
python3 main.py --config my.yaml       # use a custom config file
python3 main.py --no-api               # detection only, no web dashboard
```

---

##  Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture and frame rendering |
| `mediapipe>=0.10` | Face and hand landmark detection (Tasks API) |
| `numpy` | Numerical operations and array drawing |
| `scipy` | Euclidean distance for EAR calculation |
| `pygame` | Audio alarm generation |
| `pyyaml` | Config file parsing |
| `fastapi` | Web dashboard API |
| `uvicorn` | ASGI server for FastAPI |
| `twilio` | SMS alerts (optional) |

---

##  License

This project is open source under the [MIT License](LICENSE).

---
