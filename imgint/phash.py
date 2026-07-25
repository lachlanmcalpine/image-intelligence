"""Perceptual hashing (dHash) for near-exact duplicate detection.

Complements the embedding-based semantic dedup in pipeline.py: the SigLIP
cosine gate catches "same scene" (a static shelf, a wall), while dHash catches
"literally the same pixels" (a frozen camera, an unchanged frame) far more
cheaply -- a grayscale resize + a bit-compare, no model, no ~0.3s embed. Run
pHash first as the cheapest gate; only frames that survive it pay for an
embedding.

dHash: shrink to a tiny grayscale image and encode whether each pixel is
brighter than its right-hand neighbor. Robust to small brightness/scale
changes, sensitive to actual content change. 8x8 comparison -> 64-bit hash.
Duplicate = Hamming distance (differing bits) below a small threshold.

Pure numpy + cv2 (both already deps) -- no imagehash/PIL dependency added, same
philosophy as pq_compression.py.
"""

import cv2
import numpy as np

HASH_SIZE = 8  # 8x8 comparison grid -> 64-bit hash


def dhash(frame: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    """Return a 64-bit (for hash_size=8) perceptual hash of a BGR frame as an
    int. Same content -> same (or near-same) hash regardless of minor
    compression/exposure noise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # width is hash_size+1 so there are exactly hash_size horizontal diffs per row
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]  # (hash_size, hash_size) bool
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes -- the perceptual distance.
    0 = identical, higher = more different.
    """
    return bin(a ^ b).count("1")
