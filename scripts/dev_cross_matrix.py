"""Full cross-test: capture resolution (640x480 / 1280x720 / 1920x1080) x VAE
encode resolution (128 / 256 / 512) x compression (uncompressed fp32 vs the
shipped int8+shuffle+zlib pipeline). Measures latency, storage, and PSNR, and
saves every reconstructed image so quality can be judged visually, not just
by PSNR -- the same "show and tell" standard as the earlier VAE-resolution
comparison.

One real photo is captured at each capture resolution (same pose, taken back
to back). Each of those is then run through the codec at each encode
resolution, both with and without the production compression step.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from imgint.codec import SdxlVaeCodec, resize_short_side_and_center_crop
from imgint.compression import compress_latent, decompress_latent, dequantize_int8, quantize_int8

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out" / "cross_matrix"

CAPTURE_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
ENCODE_SIZES = [128, 256, 512]

# fixed per-image overhead outside the latent itself, matching production
EMBEDDING_BYTES = 768 * 4
NONCE_B64_BYTES = 16
ID_BYTES = 36


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def base64_len(raw_len: int) -> int:
    return ((raw_len + 2) // 3) * 4


def uncompressed_payload_bytes(latent: np.ndarray) -> int:
    ciphertext_len = latent.astype(np.float32).nbytes + 16  # AES-GCM tag, no shuffle/zlib/int8
    return EMBEDDING_BYTES + base64_len(ciphertext_len) + NONCE_B64_BYTES + 62 + ID_BYTES


def compressed_payload_bytes(compressed_len: int) -> int:
    ciphertext_len = compressed_len + 4 + 16  # +4 scale factor, +16 AES-GCM tag
    return EMBEDDING_BYTES + base64_len(ciphertext_len) + NONCE_B64_BYTES + 110 + ID_BYTES


def encode_at_size(codec: SdxlVaeCodec, image_rgb: Image.Image, size: int):
    cropped = resize_short_side_and_center_crop(image_rgb, size)
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = arr * 2 - 1
    pixel_values = codec._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(codec.device)
    with codec._torch.no_grad():
        latent_dist = codec.vae.encode(pixel_values).latent_dist
        latent = latent_dist.sample() * codec.vae.config.scaling_factor
    return latent[0].cpu().numpy().astype(np.float32), np.asarray(cropped)


def decode_latent(codec: SdxlVaeCodec, latent: np.ndarray) -> np.ndarray:
    return codec.decode(latent)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading SDXL VAE (one load, reused across the whole matrix)...")
    codec = SdxlVaeCodec()

    print("\ncapturing one real photo at each capture resolution (same pose, back to back)...")
    # MSMF can't reliably switch resolution on an already-open VideoCapture
    # (confirmed: raises a Mat assertion error mid-stream) -- open a fresh
    # capture per resolution instead, setting props before the first read.
    captures = {}
    for w, h in CAPTURE_RESOLUTIONS:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise RuntimeError("could not open camera index 1")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        frame = None
        for _ in range(15):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"could not read a frame at {w}x{h}")
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        captures[(w, h)] = image_rgb
        cap_path = OUT_DIR / f"capture_{w}x{h}.jpg"
        cv2.imwrite(str(cap_path), frame)
        print(f"  requested {w}x{h} -> actual {actual_w}x{actual_h}  saved {cap_path.relative_to(REPO_ROOT)}")

    results = []
    for cap_res in CAPTURE_RESOLUTIONS:
        image_rgb = captures[cap_res]
        cap_label = f"{cap_res[0]}x{cap_res[1]}"
        for size in ENCODE_SIZES:
            t0 = time.perf_counter()
            latent, original_crop = encode_at_size(codec, image_rgb, size)
            encode_s = time.perf_counter() - t0

            # --- uncompressed (fp32 latent, no shuffle/zlib/int8) ---
            t0 = time.perf_counter()
            recon_uncompressed = decode_latent(codec, latent)
            uncompressed_decode_ms = (time.perf_counter() - t0) * 1000
            uncompressed_bytes = uncompressed_payload_bytes(latent)
            psnr_uncompressed = psnr(original_crop, recon_uncompressed)

            # --- compressed: int8 quantize + shuffle+zlib (shipped pipeline) ---
            t0 = time.perf_counter()
            quantized, scale = quantize_int8(latent)
            compressed = compress_latent(quantized)
            compress_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            decompressed = decompress_latent(compressed, np.int8, quantized.shape)
            dequantized = dequantize_int8(decompressed, scale)
            recon_compressed = decode_latent(codec, dequantized)
            decompress_decode_ms = (time.perf_counter() - t0) * 1000
            compressed_bytes = compressed_payload_bytes(len(compressed))
            psnr_compressed = psnr(original_crop, recon_compressed)

            orig_path = OUT_DIR / f"orig_{cap_res[0]}x{cap_res[1]}_{size}.jpg"
            uncompressed_path = OUT_DIR / f"uncompressed_{cap_res[0]}x{cap_res[1]}_{size}.jpg"
            compressed_path = OUT_DIR / f"compressed_{cap_res[0]}x{cap_res[1]}_{size}.jpg"
            cv2.imwrite(str(orig_path), cv2.cvtColor(original_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(uncompressed_path), cv2.cvtColor(recon_uncompressed, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(compressed_path), cv2.cvtColor(recon_compressed, cv2.COLOR_RGB2BGR))

            row = {
                "capture_res": cap_label,
                "encode_size": size,
                "encode_ms": encode_s * 1000,
                "uncompressed_bytes": uncompressed_bytes,
                "uncompressed_psnr": psnr_uncompressed,
                "uncompressed_decode_ms": uncompressed_decode_ms,
                "compressed_bytes": compressed_bytes,
                "compressed_psnr": psnr_compressed,
                "compress_ms": compress_ms,
                "decompress_decode_ms": decompress_decode_ms,
                "orig_path": orig_path,
                "uncompressed_path": uncompressed_path,
                "compressed_path": compressed_path,
            }
            results.append(row)
            print(
                f"[{cap_label} @ {size}] encode {encode_s*1000:.0f}ms | "
                f"uncompressed {uncompressed_bytes}B PSNR {psnr_uncompressed:.2f}dB | "
                f"compressed {compressed_bytes}B PSNR {psnr_compressed:.2f}dB "
                f"(compress {compress_ms:.2f}ms)"
            )

    print("\n\n=== FULL MATRIX ===")
    header = (
        f"{'capture':>10s} {'size':>5s} {'encode(ms)':>11s} "
        f"{'uncomp(B)':>10s} {'uncomp PSNR':>12s} {'comp(B)':>9s} {'comp PSNR':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['capture_res']:>10s} {r['encode_size']:>5d} {r['encode_ms']:11.1f} "
            f"{r['uncompressed_bytes']:10d} {r['uncompressed_psnr']:12.2f} "
            f"{r['compressed_bytes']:9d} {r['compressed_psnr']:10.2f}"
        )
