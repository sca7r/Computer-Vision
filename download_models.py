"""Download the MediaPipe task model files used by the app."""

from pathlib import Path
from urllib.request import urlretrieve


MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for filename, url in MODELS.items():
        target = models_dir / filename
        if target.exists():
            print(f"[OK] {target} already exists")
            continue

        print(f"[INFO] Downloading {filename}...")
        urlretrieve(url, target)
        print(f"[OK] Saved {target}")


if __name__ == "__main__":
    main()
