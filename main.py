"""
Monitoring System — main.py
===================================
Entry point.  Starts the detection loop and optionally the FastAPI server.

Usage
-----
    python main.py                    # uses config.yaml
    python main.py --config my.yaml   # custom config file
    python main.py --no-api           # detection only, no web UI
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import distance as scipy_dist

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import load_config
from state  import state, FaceInfo
from alerts import EmailAlerter, SMSAlerter

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed — audio alarm disabled.")

_mp_hands     = mp.solutions.hands
_mp_face_mesh = mp.solutions.face_mesh

FACE_CONNECTIONS = list(_mp_face_mesh.FACEMESH_CONTOURS)
HAND_CONNECTIONS = list(_mp_hands.HAND_CONNECTIONS)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ALARM
# ══════════════════════════════════════════════════════════════════════════════

class AlarmPlayer:

    def __init__(self, cooldown: float = 5.0):
        self.cooldown  = cooldown
        self._last     = 0.0
        self._lock     = threading.Lock()
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self._sound    = self._build_beep()

    @staticmethod
    def _build_beep():
        sr   = 44100
        t    = np.linspace(0, 1, sr, endpoint=False)
        wave = np.sin(2 * np.pi * 880 * t)
        fade = np.linspace(1.0, 0.0, sr)
        return pygame.sndarray.make_sound((wave * fade * 32767).astype(np.int16))

    def trigger(self):
        now = time.time()
        with self._lock:
            if now - self._last < self.cooldown:
                return
            self._last = now
        threading.Thread(target=self._sound.play, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# EAR + PER-PERSON STATE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ear(eye) -> float:
    A = scipy_dist.euclidean(eye[1], eye[5])
    B = scipy_dist.euclidean(eye[2], eye[4])
    C = scipy_dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


class PersonState:

    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_counter = 0
        self.blinks        = 0
        self.status        = "AWAKE"

    def update(self, ear: float, threshold: float, consec: int):
        if ear < threshold:
            self.frame_counter += 1
            if self.frame_counter >= consec:
                self.status = "SLEEPING"
        else:
            if self.frame_counter >= consec:
                self.blinks += 1
            self.frame_counter = 0
            self.status = "AWAKE"


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def to_px(landmarks, w: int, h: int) -> np.ndarray:
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.int32)


def draw_face(frame, faces, w: int, h: int):
    if not faces:
        return
    for face in faces:
        pts = to_px(face, w, h)
        seg = np.array([[pts[s], pts[e]] for s, e in FACE_CONNECTIONS])
        cv2.polylines(frame, seg, False, (180, 180, 255), 1)


def draw_hands(frame, result, w: int, h: int):
    if not result or not result.hand_landmarks:
        return
    for hand in result.hand_landmarks:
        pts = to_px(hand, w, h)
        seg = np.array([[pts[s], pts[e]] for s, e in HAND_CONNECTIONS])
        cv2.polylines(frame, seg, False, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (0, 255, 0), -1)


# ══════════════════════════════════════════════════════════════════════════════
# API SERVER  (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════

def start_api(host: str, port: int):
    try:
        import uvicorn
        from api import app
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except ImportError:
        print("[ERROR] uvicorn/fastapi not installed — API disabled.")
        print("        pip install fastapi uvicorn")


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION LOOP.
# ══════════════════════════════════════════════════════════════════════════════

def run_detection(cfg, alarm, email_alerter, sms_alerter):
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]

    # Validate model files
    missing = []
    if not Path(cfg.models.face).exists():
        missing.append(cfg.models.face)
    if not cfg.detection.no_hands and not Path(cfg.models.hand).exists():
        missing.append(cfg.models.hand)
    if missing:
        print("\n[ERROR] Missing model files:")
        for f in missing:
            print(f"  {f}")
        print("\n  Run:  python download_models.py\n")
        sys.exit(1)

    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=cfg.models.face),
            num_faces=cfg.detection.max_faces,
        )
    )

    hand_landmarker = None
    if not cfg.detection.no_hands:
        hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=cfg.models.hand),
                num_hands=2,
            )
        )

    cam = cv2.VideoCapture(cfg.camera.index)
    if not cam.isOpened():
        print(f"[ERROR] Cannot open camera {cfg.camera.index}")
        sys.exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   cfg.camera.buffer_size)

    person_states = [PersonState() for _ in range(cfg.detection.max_faces)]
    executor      = ThreadPoolExecutor(max_workers=2)

    fps        = 0
    counter    = 0
    timer      = time.time()
    interval   = 1.0 / cfg.performance.fps
    prev_faces = 0
    fail_count = 0

    print(f"[INFO] Detection running — camera {cfg.camera.index}, "
          f"up to {cfg.detection.max_faces} face(s).  Press 'q' to quit.")

    try:
        while True:
            tick = time.perf_counter()

            ret, frame = cam.read()
            if not ret:
                fail_count += 1
                if fail_count > cfg.performance.max_retries:
                    print("[ERROR] Camera read failed too many times. Exiting.")
                    break
                continue
            fail_count = 0

            frame    = cv2.flip(frame, 1)
            h, w, _  = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Parallel detection
            face_future = executor.submit(face_landmarker.detect, mp_image)
            hand_future = (
                executor.submit(hand_landmarker.detect, mp_image)
                if hand_landmarker else None
            )
            face_res = face_future.result()
            hand_res = hand_future.result() if hand_future else None

            draw_face(frame,  face_res.face_landmarks, w, h)
            draw_hands(frame, hand_res, w, h)

            sleeping    = 0
            faces       = face_res.face_landmarks or []
            face_infos  = []

            for i, face in enumerate(faces):
                ps = person_states[i]

                right_eye = to_px([face[j] for j in RIGHT_EYE], w, h)
                left_eye  = to_px([face[j] for j in LEFT_EYE],  w, h)
                ear       = (calculate_ear(right_eye) + calculate_ear(left_eye)) / 2.0

                ps.update(ear, cfg.detection.ear_threshold, cfg.detection.consec_frames)
                face_infos.append(FaceInfo(index=i, ear=ear, status=ps.status, blinks=ps.blinks))

                if ps.status == "SLEEPING":
                    sleeping += 1

                    # Email / SMS alerts (fire once per sleep event with cooldown)
                    if email_alerter and email_alerter.trigger(i):
                        state.add_alert(i, "EMAIL")
                    if sms_alerter and sms_alerter.trigger(i):
                        state.add_alert(i, "SMS")

                    state.add_alert(i, "SLEEPING") if ps.frame_counter == cfg.detection.consec_frames else None

                # Per-face overlay
                xs    = [int(lm.x * w) for lm in face]
                ys    = [int(lm.y * h) for lm in face]
                x_min = max(0,  min(xs) - 10)
                y_min = max(50, min(ys) - 10)

                prefix = f"#{i + 1} " if len(faces) > 1 else ""
                color  = (0, 255, 0) if ps.status == "AWAKE" else (0, 0, 255)

                cv2.putText(frame, f"EAR {ear:.2f}",
                            (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),  2)
                cv2.putText(frame, f"{prefix}{ps.status}",
                            (x_min, y_min),      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,         2)
                cv2.putText(frame, f"Blinks {ps.blinks}",
                            (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                if ps.status == "SLEEPING":
                    cv2.putText(frame, f"ALERT: {prefix}SLEEPING!",
                                (x_min, y_min + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # Reset states for disappeared faces
            if len(faces) < prev_faces:
                for j in range(len(faces), prev_faces):
                    person_states[j].reset()
            prev_faces = len(faces)

            # Audio alarm
            if sleeping > 0 and alarm:
                alarm.trigger()

            # FPS counter
            counter += 1
            if time.time() - timer >= 1:
                fps     = counter
                counter = 0
                timer   = time.time()

            cv2.putText(frame, f"FPS {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Push frame to shared state (JPEG encode once for API stream)
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.set_frame(jpg.tobytes())
            state.update(fps, face_infos, sleeping)

            cv2.imshow("Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - tick
            if elapsed < interval:
                time.sleep(interval - elapsed)

    finally:
        state.running = False
        executor.shutdown(wait=False)
        cam.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        if hand_landmarker:
            hand_landmarker.close()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        print("[INFO] Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml", help="Path to config YAML")
    parser.add_argument("--no-api",  action="store_true",   help="Disable FastAPI server")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Alarm
    alarm = None
    if cfg.alarm.enabled and PYGAME_AVAILABLE:
        alarm = AlarmPlayer(cooldown=cfg.alarm.cooldown)
        print(f"[INFO] Audio alarm ON  (cooldown {cfg.alarm.cooldown}s)")

    # Email
    email_alerter = None
    if cfg.email.enabled:
        email_alerter = EmailAlerter(cfg.email)
        print(f"[INFO] Email alerts ON → {cfg.email.recipients}")

    # SMS
    sms_alerter = None
    if cfg.sms.enabled:
        sms_alerter = SMSAlerter(cfg.sms)
        print(f"[INFO] SMS alerts ON → {cfg.sms.to_numbers}")

    # FastAPI (background thread)
    if cfg.api.enabled and not args.no_api:
        api_thread = threading.Thread(
            target=start_api,
            args=(cfg.api.host, cfg.api.port),
            daemon=True,
        )
        api_thread.start()
        print(f"[INFO] Dashboard → http://localhost:{cfg.api.port}")

    run_detection(cfg, alarm, email_alerter, sms_alerter)


if __name__ == "__main__":
    main()