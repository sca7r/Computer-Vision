"""config.py — Loads and validates config.yaml into a typed namespace."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML not installed.  pip install pyyaml")
    sys.exit(1)


# ── Dataclasses (one per section) ─────────────────────────────────────────────

@dataclass
class CameraConfig:
    index: int       = 0
    width: int       = 640
    height: int      = 480
    buffer_size: int = 1

@dataclass
class DetectionConfig:
    ear_threshold: float = 0.25
    consec_frames: int   = 20
    max_faces: int       = 4
    no_hands: bool       = False

@dataclass
class PerformanceConfig:
    fps: int         = 30
    max_retries: int = 30

@dataclass
class ModelsConfig:
    face: str = "models/face_landmarker.task"
    hand: str = "models/hand_landmarker.task"

@dataclass
class AlarmConfig:
    enabled: bool  = True
    cooldown: float = 0.7

@dataclass
class EmailConfig:
    enabled: bool     = False
    smtp_host: str    = "smtp.gmail.com"
    smtp_port: int    = 587
    sender: str       = ""
    password: str     = ""
    recipients: List[str] = field(default_factory=list)
    cooldown: float   = 60.0

@dataclass
class SMSConfig:
    enabled: bool         = False
    twilio_sid: str       = ""
    twilio_token: str     = ""
    from_number: str      = ""
    to_numbers: List[str] = field(default_factory=list)
    cooldown: float       = 60.0

@dataclass
class APIConfig:
    enabled: bool = True
    host: str     = "0.0.0.0"
    port: int     = 8000

@dataclass
class AppConfig:
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    detection:   DetectionConfig   = field(default_factory=DetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    models:      ModelsConfig      = field(default_factory=ModelsConfig)
    alarm:       AlarmConfig       = field(default_factory=AlarmConfig)
    email:       EmailConfig       = field(default_factory=EmailConfig)
    sms:         SMSConfig         = field(default_factory=SMSConfig)
    api:         APIConfig         = field(default_factory=APIConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _from_dict(dataclass_type, data: dict):
    """Recursively populate a dataclass from a dict, ignoring unknown keys."""
    if data is None:
        return dataclass_type()
    fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in fields}
    return dataclass_type(**filtered)


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)

    if not config_path.exists():
        print(f"[WARNING] {path} not found — using defaults.")
        return AppConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig(
        camera      = _from_dict(CameraConfig,      raw.get("camera", {})),
        detection   = _from_dict(DetectionConfig,   raw.get("detection", {})),
        performance = _from_dict(PerformanceConfig, raw.get("performance", {})),
        models      = _from_dict(ModelsConfig,      raw.get("models", {})),
        alarm       = _from_dict(AlarmConfig,       raw.get("alarm", {})),
        email       = _from_dict(EmailConfig,       raw.get("email", {})),
        sms         = _from_dict(SMSConfig,         raw.get("sms", {})),
        api         = _from_dict(APIConfig,         raw.get("api", {})),
    )

    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig):
    errors = []

    if not 1 <= cfg.detection.max_faces <= 10:
        errors.append("detection.max_faces must be between 1 and 10")
    if cfg.detection.ear_threshold <= 0:
        errors.append("detection.ear_threshold must be > 0")
    if cfg.performance.fps <= 0:
        errors.append("performance.fps must be > 0")
    if cfg.email.enabled and not cfg.email.sender:
        errors.append("email.sender is required when email is enabled")
    if cfg.sms.enabled and not cfg.sms.twilio_sid:
        errors.append("sms.twilio_sid is required when SMS is enabled")

    if errors:
        print("[ERROR] Config validation failed:")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)