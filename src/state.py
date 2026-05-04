"""state.py — Thread-safe shared state between detection loop and FastAPI."""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class FaceInfo:
    index:  int
    ear:    float
    status: str
    blinks: int


@dataclass
class AlertRecord:
    timestamp: str
    face_index: int
    kind: str          # "SLEEPING" | "EMAIL" | "SMS"


class SharedState:
    """
    Single source of truth written by the detection thread,
    read by FastAPI handlers.  All access is lock-protected.
    """

    def __init__(self):
        self._lock          = threading.Lock()
        self._frame_jpg: Optional[bytes] = None
        self.fps:            int              = 0
        self.faces:          List[FaceInfo]   = []
        self.sleeping_count: int              = 0
        self.alerts:         List[AlertRecord] = []
        self.running:        bool             = True

    # ── Frame (MJPEG) ─────────────────────────────────────────────────────────

    def set_frame(self, jpg_bytes: bytes):
        with self._lock:
            self._frame_jpg = jpg_bytes

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpg

    # ── Detection stats ───────────────────────────────────────────────────────

    def update(self, fps: int, faces: List[FaceInfo], sleeping_count: int):
        with self._lock:
            self.fps            = fps
            self.faces          = faces
            self.sleeping_count = sleeping_count

    # ── Alert log ─────────────────────────────────────────────────────────────

    def add_alert(self, face_index: int, kind: str):
        record = AlertRecord(
            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            face_index = face_index,
            kind       = kind,
        )
        with self._lock:
            self.alerts.append(record)
            if len(self.alerts) > 200:           # keep last 200 only
                self.alerts = self.alerts[-200:]

    # ── Serialisable snapshots for API endpoints ──────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "fps":            self.fps,
                "face_count":     len(self.faces),
                "sleeping_count": self.sleeping_count,
                "faces": [
                    {
                        "index":  f.index,
                        "ear":    round(f.ear, 3),
                        "status": f.status,
                        "blinks": f.blinks,
                    }
                    for f in self.faces
                ],
            }

    def get_alerts(self) -> list:
        with self._lock:
            return [
                {
                    "timestamp":  a.timestamp,
                    "face_index": a.face_index,
                    "kind":       a.kind,
                }
                for a in reversed(self.alerts)
            ]


# Global singleton — imported by both main.py and api.py
state = SharedState()