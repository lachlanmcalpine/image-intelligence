import numpy as np
import pytest

from imgint.compression import (
    compress_latent,
    decompress_latent,
    dequantize_int8,
    quantize_int8,
    shuffle_bytes,
    unshuffle_bytes,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8])
def test_shuffle_unshuffle_round_trip(dtype):
    rng = np.random.default_rng(0)
    if dtype == np.int8:
        arr = rng.integers(-127, 128, size=100).astype(dtype)
    else:
        arr = rng.normal(size=100).astype(dtype)

    shuffled = shuffle_bytes(arr)
    recovered = unshuffle_bytes(shuffled, dtype, count=arr.size)

    assert np.array_equal(recovered, arr)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8])
def test_compress_decompress_latent_round_trip(dtype):
    rng = np.random.default_rng(1)
    if dtype == np.int8:
        latent = rng.integers(-127, 128, size=(4, 32, 32)).astype(dtype)
    else:
        latent = rng.normal(size=(4, 32, 32)).astype(dtype)

    compressed = compress_latent(latent)
    decompressed = decompress_latent(compressed, dtype, latent.shape)

    assert np.array_equal(decompressed, latent)


def test_compression_actually_shrinks_a_real_shaped_latent():
    # smooth, spatially-correlated data (like a real VAE latent), not pure
    # noise -- shuffle+zlib should beat the raw byte size
    rng = np.random.default_rng(2)
    base = rng.normal(size=(4, 32, 32)).astype(np.float32)
    smooth = np.repeat(np.repeat(base, 4, axis=1), 4, axis=2)[:, :32, :32].astype(np.float16)

    compressed = compress_latent(smooth)
    assert len(compressed) < smooth.nbytes


def test_quantize_dequantize_is_close_but_not_exact():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(4, 32, 32)).astype(np.float32)

    quantized, scale = quantize_int8(latent)
    recovered = dequantize_int8(quantized, scale)

    assert quantized.dtype == np.int8
    assert not np.array_equal(recovered, latent)  # lossy
    assert np.allclose(recovered, latent, atol=scale)  # within one quantization step


def test_quantize_handles_all_zero_latent():
    latent = np.zeros((4, 32, 32), dtype=np.float32)
    quantized, scale = quantize_int8(latent)
    assert scale == 1.0  # avoids division by zero
    assert np.array_equal(dequantize_int8(quantized, scale), latent)


def test_quantize_scale_maps_max_value_near_int8_range():
    latent = np.array([-3.0, 0.0, 1.5, 3.0], dtype=np.float32)
    quantized, scale = quantize_int8(latent)
    assert quantized.min() >= -127
    assert quantized.max() <= 127
    # the max-magnitude value should land at or near the int8 boundary
    assert abs(int(quantized[np.argmax(np.abs(latent))])) >= 120
