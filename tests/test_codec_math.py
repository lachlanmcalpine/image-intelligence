import numpy as np
import pytest
import torch

from imgint.codec import SdxlVaeCodec


class _FakeConfig:
    def __init__(self, latent_channels, scaling_factor):
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor


class _FakeLatentDist:
    def __init__(self, tensor):
        self._tensor = tensor

    def sample(self):
        return self._tensor


class _FakeDecodeOutput:
    def __init__(self, tensor):
        self.sample = tensor


class _FakeVae:
    """Identity-ish stand-in VAE: encode always returns a fixed raw latent
    (0.5 everywhere), and decode records whatever it was handed -- lets us
    verify SdxlVaeCodec's own scaling/shape logic without a real model.
    """

    def __init__(self, latent_channels=4, scaling_factor=0.13025):
        self.config = _FakeConfig(latent_channels, scaling_factor)
        self.last_decode_input = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, pixel_values):
        b, _c, h, w = pixel_values.shape
        raw = torch.full((b, self.config.latent_channels, h // 8, w // 8), 0.5)
        return type("Enc", (), {"latent_dist": _FakeLatentDist(raw)})()

    def decode(self, latents):
        self.last_decode_input = latents.clone()
        b, _c, h, w = latents.shape
        img = torch.zeros((b, 3, h * 8, w * 8))
        return _FakeDecodeOutput(img)


def test_scaling_factor_applied_symmetrically():
    fake_vae = _FakeVae(scaling_factor=2.0)
    codec = SdxlVaeCodec(vae=fake_vae)

    pixel_values = np.zeros((32, 32, 3), dtype=np.uint8)
    latent = codec.encode(pixel_values)

    # raw latent from the fake vae is 0.5; encode() scales it by scaling_factor=2.0
    assert np.allclose(latent, 1.0)

    codec.decode(latent)

    # decode() must divide the stored (scaled) latent back down by the same
    # scaling_factor before handing it to vae.decode. If that division were
    # missing (or applied twice), vae.decode would see 1.0 or 0.25 instead of
    # the original raw 0.5 -- exactly the asymmetric-scaling bug this guards.
    assert torch.allclose(fake_vae.last_decode_input, torch.full_like(fake_vae.last_decode_input, 0.5))


def test_decode_output_shape_matches_upscaled_latent():
    fake_vae = _FakeVae(scaling_factor=1.0)
    codec = SdxlVaeCodec(vae=fake_vae)

    # encode() always resizes to the fixed TARGET_SIZE (256) first, regardless
    # of input size, so per-frame latent/storage size is predictable.
    pixel_values = np.zeros((64, 96, 3), dtype=np.uint8)
    latent = codec.encode(pixel_values)
    assert latent.shape == (4, 32, 32)  # 256 / 8 = 32

    img = codec.decode(latent)
    assert img.shape == (256, 256, 3)  # 32 * 8 = 256
    assert img.dtype == np.uint8


def test_rejects_wrong_channel_count():
    fake_vae = _FakeVae(latent_channels=16)  # FLUX-shaped, wrong VAE family
    with pytest.raises(ValueError, match="16"):
        SdxlVaeCodec(vae=fake_vae)


def test_decode_rejects_wrong_channel_latent():
    fake_vae = _FakeVae(latent_channels=4)
    codec = SdxlVaeCodec(vae=fake_vae)
    bad_latent = np.zeros((16, 4, 4), dtype=np.float32)  # FLUX-shaped latent
    with pytest.raises(ValueError, match="16"):
        codec.decode(bad_latent)
