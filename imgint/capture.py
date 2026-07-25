"""Webcam frame capture."""

import cv2
import numpy as np

# Requested at open time, before the first read -- MSMF can't reliably switch
# resolution on an already-streaming capture (confirmed: raises a Mat
# assertion error mid-stream). Chosen over the driver's 640x480 default after
# directly comparing reconstructions: 1920x1080 gave visibly sharper detail
# at every VAE encode size tested, at the cost of a wider (16:9) crop -- see
# todo.md.
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
        frame = None
        for _ in range(max(warmup_frames, 1)):
            ok, frame = cap.read()
            if not ok:
                raise CameraError(f"could not read a frame from camera index {index}")
        return frame
    finally:
        cap.release()


class Camera:
    """Keeps the camera open across many reads -- for repeated/real-time
    capture, opening+warming up fresh every time (as capture_frame() does)
    dominates latency. Warmup happens once, at construction.
    """

    def __init__(self, index: int = 0, warmup_frames: int = 10):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraError(f"could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
        for _ in range(max(warmup_frames, 1)):
            ok, _ = self.cap.read()
            if not ok:
                self.cap.release()
                raise CameraError(f"could not read a frame from camera index {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise CameraError("could not read a frame")
        return frame

    def release(self) -> None:
        self.cap.release()
