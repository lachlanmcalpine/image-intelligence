import cv2
import numpy as np

from imgint.phash import dhash, hamming


def _patterned_frame(seed: int, h: int = 128, w: int = 128) -> np.ndarray:
    """A frame with genuine per-seed structure: a random 8x8 block pattern
    upscaled. dHash needs real spatial variation to distinguish content -- a
    flat fill or a shared gradient would hash the same regardless of seed.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    gray = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.stack([gray, gray, gray], axis=2)  # BGR, 3 channels


def test_identical_frames_hash_identically():
    frame = _patterned_frame(0)
    assert hamming(dhash(frame), dhash(frame.copy())) == 0


def test_hamming_is_symmetric_and_zero_on_self():
    a = dhash(_patterned_frame(1))
    b = dhash(_patterned_frame(2))
    assert hamming(a, a) == 0
    assert hamming(a, b) == hamming(b, a)


def test_hash_fits_in_64_bits():
    value = dhash(_patterned_frame(3))
    assert 0 <= value < 2**64


def test_near_identical_closer_than_different():
    frame = _patterned_frame(4)
    # tiny brightness bump -> should stay perceptually near-identical
    nudged = np.clip(frame.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    very_different = _patterned_frame(999)

    near = hamming(dhash(frame), dhash(nudged))
    far = hamming(dhash(frame), dhash(very_different))

    assert near < far
    assert near <= 4  # within the production threshold


def test_resolution_change_stays_below_scene_change():
    """A resolution change of the same content must hash much closer than a
    genuinely different scene -- this is the property the production threshold
    relies on (pHash catches frozen/near-identical frames, semantic dedup
    handles same-scene-different-composition).
    """
    frame = _patterned_frame(7, h=240, w=320)
    downscaled = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    different = _patterned_frame(1234, h=240, w=320)

    same_content = hamming(dhash(frame), dhash(downscaled))
    diff_content = hamming(dhash(frame), dhash(different))

    assert same_content < diff_content
