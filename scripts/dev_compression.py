"""Measure the real effect of shuffle+zlib pre-compression (and, as a second
prototype, int8 quantization) on a real VAE latent from a live capture.
Reports, for each option: latent bytes, full per-image cloud payload bytes
(embedding + encrypted-latent-base64 + metadata), additional-loss PSNR
(relative to the fp32-latent reconstruction, isolating what each compression
step costs on top of the VAE's own lossy-ness), and processing time for the
encode-side and decode-side steps (averaged over repeated runs since these
are fast in-memory operations).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import cv2

from imgint.capture import capture_frame
from imgint.codec import SdxlVaeCodec, TARGET_SIZE, resize_short_side_and_center_crop
from imgint.compression import compress_latent, decompress_latent
from imgint.crypto import encrypt_latent, generate_aes_key
from PIL import Image

N_TIMING_RUNS = 50

# fixed per-image overhead outside the latent itself (measured earlier from a
# real stored record): embedding (768-dim fp32) + nonce (base64) + metadata + id
EMBEDDING_BYTES = 768 * 4
NONCE_B64_BYTES = 16
ID_BYTES = 36


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def quantize_int8(latent: np.ndarray) -> tuple:
    """Simple per-tensor symmetric quantization: scale so the max abs value
    maps to 127, round to int8. Returns (quantized, scale) -- scale is needed
    to dequantize back to float.
    """
    scale = np.abs(latent).max() / 127.0
    if scale == 0:
        scale = 1.0
    quantized = np.round(latent / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale


def time_it(fn, n=N_TIMING_RUNS):
    start = time.perf_counter()
    for _ in range(n):
        result = fn()
    elapsed = time.perf_counter() - start
    return result, (elapsed / n) * 1000  # ms per call


def base64_len(raw_len: int) -> int:
    return ((raw_len + 2) // 3) * 4


def full_payload_bytes(latent_bytes_len: int, extra_metadata_bytes: int) -> int:
    ciphertext_len = latent_bytes_len + 16  # AES-GCM tag
    return (
        EMBEDDING_BYTES
        + base64_len(ciphertext_len)
        + NONCE_B64_BYTES
        + extra_metadata_bytes
        + ID_BYTES
    )


if __name__ == "__main__":
    print("capturing a real frame and encoding it...")
    frame = capture_frame(index=1)
    codec = SdxlVaeCodec()
    aes_key = generate_aes_key()

    t0 = time.perf_counter()
    latent = codec.encode(frame)  # fp32, shape (4, 32, 32)
    encode_ms = (time.perf_counter() - t0) * 1000
    original_image = codec.decode(latent)

    # the real captured photo, cropped the same way the codec does internally,
    # for a true end-to-end quality comparison (isolates total loss including
    # the VAE's own lossy-ness, not just the marginal cost of each compression
    # step on top of the fp32-latent reconstruction)
    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    real_photo_256 = np.asarray(
        resize_short_side_and_center_crop(Image.fromarray(original_rgb), TARGET_SIZE)
    )

    rows = []

    # --- baseline: fp32 raw ---
    fp32_bytes = latent.tobytes()
    rows.append(
        {
            "name": "fp32 (baseline)",
            "latent_bytes": len(fp32_bytes),
            "full_payload": full_payload_bytes(len(fp32_bytes), 62),
            "psnr": float("inf"),
            "psnr_vs_photo": psnr(real_photo_256, original_image),
            "compress_ms": 0.0,
            "decompress_ms": 0.0,
        }
    )

    # --- current: fp16 raw (what we ship today) ---
    _, cast_ms = time_it(lambda: latent.astype(np.float16))
    latent_fp16 = latent.astype(np.float16)
    fp16_bytes = latent_fp16.tobytes()
    _, decode_cast_ms = time_it(lambda: latent_fp16.astype(np.float32))
    recon_fp16 = codec.decode(latent_fp16.astype(np.float32))
    rows.append(
        {
            "name": "fp16 (current)",
            "latent_bytes": len(fp16_bytes),
            "full_payload": full_payload_bytes(len(fp16_bytes), 62),
            "psnr": psnr(original_image, recon_fp16),
            "psnr_vs_photo": psnr(real_photo_256, recon_fp16),
            "compress_ms": cast_ms,
            "decompress_ms": decode_cast_ms,
        }
    )

    # --- option 1: fp16 + shuffle + zlib (lossless on top of fp16) ---
    compressed, compress_ms = time_it(lambda: compress_latent(latent_fp16))
    decompressed, decompress_ms = time_it(
        lambda: decompress_latent(compressed, np.float16, latent_fp16.shape)
    )
    assert np.array_equal(decompressed, latent_fp16), "shuffle+zlib round-trip must be exact"
    recon_shuffle = codec.decode(decompressed.astype(np.float32))
    rows.append(
        {
            "name": "fp16 + shuffle+zlib (option 1)",
            "latent_bytes": len(compressed),
            "full_payload": full_payload_bytes(len(compressed), 85),
            "psnr": psnr(original_image, recon_shuffle),
            "psnr_vs_photo": psnr(real_photo_256, recon_shuffle),
            "compress_ms": compress_ms,
            "decompress_ms": decompress_ms,
        }
    )

    # --- option 2: int8 quantization + shuffle + zlib ---
    (quantized, scale), quantize_ms = time_it(lambda: quantize_int8(latent))
    compressed_int8, compress_int8_ms = time_it(lambda: compress_latent(quantized))
    decompressed_int8, decompress_int8_ms = time_it(
        lambda: decompress_latent(compressed_int8, np.int8, quantized.shape)
    )
    assert np.array_equal(decompressed_int8, quantized), "shuffle+zlib round-trip must be exact"
    _, dequantize_ms = time_it(lambda: dequantize_int8(decompressed_int8, scale))
    dequantized = dequantize_int8(decompressed_int8, scale)
    recon_int8 = codec.decode(dequantized)
    rows.append(
        {
            "name": "int8 + shuffle+zlib (option 2, prototype)",
            "latent_bytes": len(compressed_int8) + 4,  # +4 bytes to carry the scale factor
            "full_payload": full_payload_bytes(len(compressed_int8) + 4, 110),
            "psnr": psnr(original_image, recon_int8),
            "psnr_vs_photo": psnr(real_photo_256, recon_int8),
            "compress_ms": quantize_ms + compress_int8_ms,
            "decompress_ms": decompress_int8_ms + dequantize_ms,
        }
    )

    print(f"\nVAE encode (fp32, not part of compression cost): {encode_ms:.2f} ms\n")

    header = (
        f"{'method':45s} {'latent B':>9s} {'full payload':>13s} "
        f"{'PSNR vs fp32-VAE':>17s} {'PSNR vs real photo':>19s} {'encode ms':>10s} {'decode ms':>10s}"
    )
    print(header)
    print("-" * len(header))
    baseline = rows[0]["full_payload"]
    for r in rows:
        pct = 100 * r["full_payload"] / baseline
        psnr_str = "lossless" if r["psnr"] == float("inf") else f"{r['psnr']:.2f}"
        print(
            f"{r['name']:45s} {r['latent_bytes']:9d} "
            f"{r['full_payload']:6d}B ({pct:4.1f}%) {psnr_str:>17s} "
            f"{r['psnr_vs_photo']:19.2f} "
            f"{r['compress_ms']:9.4f} {r['decompress_ms']:9.4f}"
        )
