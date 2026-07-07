"""Milestone 3 verification: encode a real captured frame through the SDXL
VAE, decode it back, save original vs. reconstructed side by side, and print
PSNR so we have a number as well as an eyeball check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from imgint.codec import SdxlVaeCodec, TARGET_SIZE, resize_short_side_and_center_crop
from imgint.capture import capture_frame
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)

    print("capturing a frame from camera index 1...")
    frame_bgr = capture_frame(index=1)

    print("loading SDXL VAE (madebyollin/sdxl-vae-fp16-fix, first run downloads ~335 MB)...")
    codec = SdxlVaeCodec()

    latent = codec.encode(frame_bgr)
    print(f"latent shape: {latent.shape}, dtype: {latent.dtype}, bytes: {latent.nbytes}")

    reconstructed = codec.decode(latent)  # RGB uint8 (256, 256, 3)

    # build the same 256x256 center-cropped original for a fair pixel comparison
    original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_256 = np.asarray(
        resize_short_side_and_center_crop(Image.fromarray(original_rgb), TARGET_SIZE)
    )

    score = psnr(original_256, reconstructed)
    print(f"PSNR (original vs. reconstructed, both 256x256): {score:.2f} dB")

    side_by_side = np.concatenate([original_256, reconstructed], axis=1)
    out_path = out_dir / "vae_roundtrip.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"saved original|reconstructed side by side -> {out_path}")
