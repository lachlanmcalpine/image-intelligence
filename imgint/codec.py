"""VAE encode/decode for compressed image latents (pure autoencoder
reconstruction -- no diffusion, so no hallucinated content).

Uses the SDXL VAE family (4 latent channels, 8x spatial downsample) via a
dedicated VAE-only repo, NOT FLUX's VAE (16 channels) -- we never use FLUX's
diffusion transformer, so its VAE offers no benefit here and would quietly
quadruple per-frame storage.
"""

from typing import Protocol

import numpy as np

EXPECTED_LATENT_CHANNELS = 4  # SDXL-family. FLUX's VAE has 16 channels -- if
# that ever shows up here, the wrong VAE family got loaded.

TARGET_SIZE = 128  # fixed encode resolution so latent/storage size is predictable --
# chosen over 256/512 for latency+storage on this CPU-only machine; see todo.md


class VAECodec(Protocol):
    def encode(self, image) -> np.ndarray: ...
    def decode(self, latent: np.ndarray) -> np.ndarray: ...


class SdxlVaeCodec:
    """Wraps madebyollin/sdxl-vae-fp16-fix -- a repo containing *only* the VAE
    (no UNet/text encoder), so from_pretrained can't accidentally pull a
    multi-GB SDXL pipeline.

    Accepts a pre-built `vae` object for dependency injection in tests (see
    tests/test_codec_math.py) so the scaling-factor/channel-count logic can be
    verified without downloading or running a real model.
    """

    MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

    def __init__(self, vae=None, model_id: str | None = None, device: str = "cpu"):
        import torch

        self._torch = torch
        self.device = device
        if vae is None:
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(model_id or self.MODEL_ID).to(device).eval()
        self.vae = vae
        if self.vae.config.latent_channels != EXPECTED_LATENT_CHANNELS:
            raise ValueError(
                f"expected a {EXPECTED_LATENT_CHANNELS}-channel VAE (SDXL-family), "
                f"got {self.vae.config.latent_channels} channels -- wrong VAE family loaded"
            )

    def encode(self, image) -> np.ndarray:
        """image: PIL.Image or BGR ndarray. Returns the latent as a numpy
        array, scaled by vae.config.scaling_factor for storage -- undone
        symmetrically in decode() below (never apply this on only one side;
        that silently corrupts every reconstruction).
        """
        pixel_values = self._to_tensor(image)
        with self._torch.no_grad():
            latent_dist = self.vae.encode(pixel_values).latent_dist
            latent = latent_dist.sample() * self.vae.config.scaling_factor
        return latent[0].cpu().numpy().astype(np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Inverse of encode(): returns an RGB uint8 ndarray (H, W, 3)."""
        if latent.shape[0] != EXPECTED_LATENT_CHANNELS:
            raise ValueError(
                f"expected a {EXPECTED_LATENT_CHANNELS}-channel latent, got shape {latent.shape}"
            )
        t = self._torch.from_numpy(latent).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            decoded = self.vae.decode(t / self.vae.config.scaling_factor).sample[0]
        decoded = (decoded.clamp(-1, 1) + 1) / 2 * 255
        return decoded.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def _to_tensor(self, image):
        import cv2
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = resize_short_side_and_center_crop(image.convert("RGB"), TARGET_SIZE)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = arr * 2 - 1  # VAE expects [-1, 1]
        return self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)


def resize_short_side_and_center_crop(image, size: int):
    """Resize preserving aspect ratio (short side -> size), then center-crop
    to size x size -- NOT a plain square resize, which would squash aspect
    ratio and add geometric distortion unrelated to compression loss.
    """
    from PIL import Image

    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return image.crop((left, top, left + size, top + size))
