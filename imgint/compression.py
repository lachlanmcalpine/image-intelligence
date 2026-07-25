"""Lossless pre-compression for VAE latents, applied BEFORE encryption --
encrypted ciphertext is high-entropy by design and has nothing left to
compress, so this only works pre-encryption (compress-then-encrypt).

Byte-shuffling regroups a float array's bytes by position (all byte-0s, then
all byte-1s, ...) instead of value (value-0's bytes, then value-1's bytes,
...). VAE latents are spatially smooth with similar-magnitude neighboring
values, so the high-order bytes (sign/exponent) repeat far more often once
grouped together this way -- exactly the trick HDF5/Blosc use for scientific
float arrays. zlib alone barely touches this data; shuffle-then-zlib does
much better because the repetition is now byte-aligned instead of scattered
one value-width apart.
"""

import zlib

import numpy as np


def shuffle_bytes(arr: np.ndarray) -> bytes:
    """Reorder an array's raw bytes from [v0b0 v0b1 v1b0 v1b1 ...] to
    [v0b0 v1b0 ... v0b1 v1b1 ...] (byte-plane order). Reversible via
    unshuffle_bytes given the same dtype and element count.
    """
    itemsize = arr.dtype.itemsize
    raw = np.frombuffer(arr.tobytes(), dtype=np.uint8).reshape(-1, itemsize)
    return raw.T.tobytes()


def unshuffle_bytes(data: bytes, dtype: np.dtype, count: int) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    raw = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, count)
    return np.frombuffer(raw.T.tobytes(), dtype=dtype)


def compress_latent(latent: np.ndarray) -> bytes:
    """Shuffle + zlib-compress a latent array for storage. Lossless --
    decompress_latent recovers the exact input bytes.
    """
    shuffled = shuffle_bytes(latent)
    return zlib.compress(shuffled, level=9)


def decompress_latent(data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    shuffled = zlib.decompress(data)
    count = int(np.prod(shape))
    flat = unshuffle_bytes(shuffled, dtype, count)
    return flat.reshape(shape)


def quantize_int8(latent: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization: scale so the max abs value maps to
    127, round to int8. Lossy -- `scale` must travel alongside the data (as
    metadata) to dequantize. Measured to cost ~0.07 dB of end-to-end
    reconstruction quality (31.90 vs 31.97 dB against the real photo) because
    the VAE's own 256x256->4x32x32 compression already dominates the error --
    see docs/storage-compression.md.
    """
    scale = float(np.abs(latent).max() / 127.0)
    if scale == 0:
        scale = 1.0
    quantized = np.round(latent / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale
