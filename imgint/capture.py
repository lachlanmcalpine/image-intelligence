"""Webcam frame capture. Correctness-first: no threading, no real-time
buffering yet — grab-and-release per call, or a simple sleep-interval loop.
"""

import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def capture_frame(index: int = 0, warmup_frames: int = 10) -> np.ndarray:
    """Open the camera, discard a few warm-up frames (auto-exposure/gain need
    a moment to settle, and the very first frame(s) are often black), grab a
    real BGR frame, and release it.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CameraError(f"could not open camera index {index}")
    try:
        frame = None
        for _ in range(max(warmup_frames, 1)):
            ok, frame = cap.read()
            if not ok:
                raise CameraError(f"could not read a frame from camera index {index}")
        return frame
    finally:
        cap.release()


def capture_loop(on_frame, interval_s: float = 2.0, index: int = 0, max_frames: int | None = None):
    """Repeatedly capture a frame every `interval_s` seconds and call `on_frame(frame)`.

    Keeps the camera open for the duration of the loop (unlike capture_frame,
    which opens/releases per call) so repeated captures don't pay the camera
    open/warm-up cost every time.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CameraError(f"could not open camera index {index}")
    count = 0
    try:
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                raise CameraError(f"could not read a frame from camera index {index}")
            on_frame(frame)
            count += 1
            if max_frames is None or count < max_frames:
                time.sleep(interval_s)
    finally:
        cap.release()
