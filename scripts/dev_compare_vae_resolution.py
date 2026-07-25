"""Phase 3b: VAE encode resolution -- 128^2 vs 256^2 (current, TARGET_SIZE in
imgint/codec.py) vs 512^2 -- measured on the same real captured photo.

Does NOT change TARGET_SIZE; reuses resize_short_side_and_center_crop() at
each size with the already-loaded VAE, so this is a single model load, three
encode/decode passes. Runs the *same* production compression pipeline
(quantize_int8 + compress_latent) at each resolution for a fair storage
comparison, and PSNR is computed against a same-resolution crop of the real
photo (like-for-like, not against the 256^2 version for all three).

Saves original + reconstructed images for each resolution to out/vae_res/
so they can be visually compared (PSNR alone isn't enough -- see the
generated HTML comparison this script's caller builds).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from imgint.codec import SdxlVaeCodec, resize_short_side_and_center_crop
from imgint.compression import compress_latent, quantize_int8

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTO_PATH = REPO_ROOT / "static" / "captures" / "c9b9a747e5324675aac8c72716f53293.jpg"
OUT_DIR = REPO_ROOT / "out" / "vae_res"

SIZES = [128, 256, 512]
N_TIMING_RUNS = 5

# fixed per-image overhead outside the latent itself, matching production
# (embedding + nonce b64 + metadata + id) -- doesn't vary with VAE resolution
EMBEDDING_BYTES = 768 * 4
NONCE_B64_BYTES = 16
METADATA_BYTES = 110
ID_BYTES = 36


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def base64_len(raw_len: int) -> int:
    return ((raw_len + 2) // 3) * 4


def full_payload_bytes(compressed_latent_len: int) -> int:
    ciphertext_len = compressed_latent_len + 4 + 16  # +4 scale factor, +16 AES-GCM tag
    return EMBEDDING_BYTES + base64_len(ciphertext_len) + NONCE_B64_BYTES + METADATA_BYTES + ID_BYTES


def encode_at_size(codec: SdxlVaeCodec, image_rgb: Image.Image, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors SdxlVaeCodec.encode() but with a parameterized target size
    instead of the hardcoded TARGET_SIZE, reusing the already-loaded VAE.
    Returns (latent, cropped_original_rgb_array).
    """
    cropped = resize_short_side_and_center_crop(image_rgb, size)
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = arr * 2 - 1
    pixel_values = codec._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(codec.device)
    with codec._torch.no_grad():
        latent_dist = codec.vae.encode(pixel_values).latent_dist
        latent = latent_dist.sample() * codec.vae.config.scaling_factor
    return latent[0].cpu().numpy().astype(np.float32), np.asarray(cropped)


if __name__ == "__main__":
    if not PHOTO_PATH.exists():
        print(f"missing expected real photo: {PHOTO_PATH}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading SDXL VAE (one load, reused across all 3 resolutions)...")
    codec = SdxlVaeCodec()

    frame_bgr = cv2.imread(str(PHOTO_PATH))
    image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

    results = []
    for size in SIZES:
        print(f"\n=== {size}x{size} ===")

        # steady-state encode latency
        times = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            latent, original_crop = encode_at_size(codec, image_rgb, size)
            times.append(time.perf_counter() - t0)
        steady = times[1:]
        avg_encode_s = sum(steady) / len(steady)
        print(f"  encode latency (5 runs): {[f'{t*1000:.0f}ms' for t in times]}")
        print(f"  steady-state avg (excl. 1st): {avg_encode_s*1000:.1f}ms")

        print(f"  latent shape: {latent.shape}")

        quantized, scale = quantize_int8(latent)
        compressed = compress_latent(quantized)
        payload_bytes = full_payload_bytes(len(compressed))
        print(f"  compressed latent bytes: {len(compressed)}")
        print(f"  full per-image cloud payload (int8+shuffle+zlib+AES-GCM+base64+embedding): {payload_bytes} bytes")

        reconstructed = codec.decode(latent)
        score = psnr(original_crop, reconstructed)
        print(f"  PSNR (vs same-resolution real photo crop): {score:.2f} dB")

        orig_path = OUT_DIR / f"original_{size}.jpg"
        recon_path = OUT_DIR / f"reconstructed_{size}.jpg"
        cv2.imwrite(str(orig_path), cv2.cvtColor(original_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(recon_path), cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print(f"  saved: {orig_path.relative_to(REPO_ROOT)}, {recon_path.relative_to(REPO_ROOT)}")

        results.append(
            {
                "size": size,
                "encode_ms": avg_encode_s * 1000,
                "latent_shape": latent.shape,
                "compressed_bytes": len(compressed),
                "payload_bytes": payload_bytes,
                "psnr": score,
                "original_path": orig_path,
                "reconstructed_path": recon_path,
            }
        )

    print("\n\n=== SUMMARY ===")
    header = f"{'size':>6s} {'encode(ms)':>11s} {'latent':>14s} {'payload(B)':>11s} {'PSNR(dB)':>9s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['size']:>6d} {r['encode_ms']:11.1f} {str(r['latent_shape']):>14s} "
            f"{r['payload_bytes']:11d} {r['psnr']:9.2f}"
        )
