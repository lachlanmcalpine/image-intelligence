"""Milestone 1 verification: grab one frame from the webcam and save it so we
can visually confirm the camera actually works on this machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from imgint.capture import capture_frame

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)

    frame = capture_frame(index=1)
    out_path = out_dir / "frame_0001.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"captured frame shape={frame.shape} -> {out_path}")
