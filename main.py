"""
Driver Monitoring System
========================

Hand & Face Tracking with Sleep Detection using MediaPipe Tasks API.

Features
--------
✓ Face + hand landmark tracking
✓ Eye Aspect Ratio sleep detection
✓ Multi-person support
✓ Real-time FPS counter
✓ Audio alarm with cooldown
✓ CLI configuration
✓ Parallel detection execution

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


cv2.setUseOptimized(True)
cv2.setNumThreads(4)


try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed. Alarm disabled.")


_mp_hands     = mp.solutions.hands
_mp_face_mesh = mp.solutions.face_mesh

FACE_CONNECTIONS = list(_mp_face_mesh.FACEMESH_CONTOURS)
HAND_CONNECTIONS = list(_mp_hands.HAND_CONNECTIONS)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",         type=int,   default=0)
    parser.add_argument("--ear-threshold",  type=float, default=0.25)
    parser.add_argument("--consec-frames",  type=int,   default=20)
    parser.add_argument("--max-faces",      type=int,   default=4)
    parser.add_argument("--fps",            type=int,   default=30)
    parser.add_argument("--no-alarm",       action="store_true")
    parser.add_argument("--alarm-cooldown", type=float, default=1)
    parser.add_argument("--no-hands",       action="store_true")
    parser.add_argument("--face-model",     default="models/face_landmarker.task")
    parser.add_argument("--hand-model",     default="models/hand_landmarker.task")
    parser.add_argument("--max-retries",    type=int,   default=30)
    return parser.parse_args()


def check_models(args):
    missing = []
    if not Path(args.face_model).exists():
        missing.append(args.face_model)
    if not args.no_hands and not Path(args.hand_model).exists():
        missing.append(args.hand_model)
    if missing:
        print("\n[ERROR] Missing model files:")
        for f in missing:
            print(f"  {f}")
        print("\n  Run:  python download_models.py\n")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ALARM  — fade-out envelope to remove audible click
# ══════════════════════════════════════════════════════════════════════════════

class AlarmPlayer:

    def __init__(self, cooldown=5.0):
        self.cooldown  = cooldown
        self.last_time = 0
        self.lock      = threading.Lock()

        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self.sound = self._build_beep()

    def _build_beep(self):
        sr   = 44100
        t    = np.linspace(0, 1, sr, endpoint=False)
        wave = np.sin(2 * np.pi * 880 * t)
        # fade-out to avoid the harsh click at end of beep
        fade = np.linspace(1.0, 0.0, sr)
        wave = (wave * fade * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def trigger(self):
        now = time.time()
        with self.lock:
            if now - self.last_time < self.cooldown:
                return
            self.last_time = now
        threading.Thread(target=self.sound.play, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# EAR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ear(eye):
    A = scipy_dist.euclidean(eye[1], eye[5])
    B = scipy_dist.euclidean(eye[2], eye[4])
    C = scipy_dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ══════════════════════════════════════════════════════════════════════════════
# PER-PERSON STATE
# ══════════════════════════════════════════════════════════════════════════════

class PersonState:

    def __init__(self):
        self.reset()

    def reset(self):
        self.frame_counter = 0
        self.blinks        = 0
        self.status        = "AWAKE"

    def update(self, ear, threshold, consec):
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

def to_px(landmarks, w, h):
    coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    return coords.astype(np.int32)


def draw_face(frame, faces, w, h):
    if not faces:
        return
    for face in faces:
        pts = to_px(face, w, h)
        seg = np.array([[pts[s], pts[e]] for s, e in FACE_CONNECTIONS])
        cv2.polylines(frame, seg, False, (180, 180, 255), 1)


def draw_hands(frame, result, w, h):
    if not result or not result.hand_landmarks:
        return
    for hand in result.hand_landmarks:
        pts = to_px(hand, w, h)
        seg = np.array([[pts[s], pts[e]] for s, e in HAND_CONNECTIONS])
        cv2.polylines(frame, seg, False, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (0, 255, 0), -1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    args = parse_args()
    check_models(args)

    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]

    alarm = None
    if not args.no_alarm and PYGAME_AVAILABLE:
        alarm = AlarmPlayer(args.alarm_cooldown)

    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=args.face_model),
            num_faces=args.max_faces,
        )
    )

    hand_landmarker = None
    if not args.no_hands:
        hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=args.hand_model),
                num_hands=2,
            )
        )

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print("[ERROR] Camera not accessible")
        sys.exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    states     = [PersonState() for _ in range(args.max_faces)]
    executor   = ThreadPoolExecutor(max_workers=2)

    fps      = 0
    counter  = 0
    timer    = time.time()
    interval = 1.0 / args.fps

    # initialise properly so reset logic works on first frame
    prev_faces = 0
    fail_count = 0

    #  try/finally guarantees cleanup on any exception
    try:
        while True:
            start = time.perf_counter()

            ret, frame = cam.read()
            if not ret:
                fail_count += 1
                if fail_count > args.max_retries:
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
            hand_future = executor.submit(hand_landmarker.detect, mp_image) if hand_landmarker else None

            face_res = face_future.result()
            hand_res = hand_future.result() if hand_future else None

            draw_face(frame,  face_res.face_landmarks, w, h)
            draw_hands(frame, hand_res, w, h)

            sleeping = 0
            faces    = face_res.face_landmarks or []

            for i, face in enumerate(faces):
                state = states[i]

                right_eye = to_px([face[j] for j in RIGHT_EYE], w, h)
                left_eye  = to_px([face[j] for j in LEFT_EYE],  w, h)
                ear       = (calculate_ear(right_eye) + calculate_ear(left_eye)) / 2.0

                state.update(ear, args.ear_threshold, args.consec_frames)

                if state.status == "SLEEPING":
                    sleeping += 1

                # anchor labels to bounding box top-left, not nose tip (face[0])
                xs    = [int(lm.x * w) for lm in face]
                ys    = [int(lm.y * h) for lm in face]
                x_min = max(0,  min(xs) - 10)
                y_min = max(50, min(ys) - 10)

                prefix = f"#{i + 1} " if len(faces) > 1 else ""
                color  = (0, 255, 0) if state.status == "AWAKE" else (0, 0, 255)

                cv2.putText(frame, f"EAR {ear:.2f}",
                            (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),  2)
                cv2.putText(frame, f"{prefix}{state.status}",
                            (x_min, y_min),      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,         2)
                cv2.putText(frame, f"Blinks {state.blinks}",
                            (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # estore the SLEEPING alert banner
                if state.status == "SLEEPING":
                    cv2.putText(frame, f"ALERT: {prefix}SLEEPING!",
                                (x_min, y_min + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # reset state for faces that disappeared this frame
            if len(faces) < prev_faces:
                for j in range(len(faces), prev_faces):
                    states[j].reset()
            prev_faces = len(faces)

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

            cv2.imshow("Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    # always release resources, even on crash
    finally:
        executor.shutdown(wait=False)
        cam.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        if hand_landmarker:
            hand_landmarker.close()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    run()