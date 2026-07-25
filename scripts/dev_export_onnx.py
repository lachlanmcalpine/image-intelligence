"""Export SigLIP (image + text towers) and the VAE encoder to ONNX, then
verify each exported graph produces output numerically close to the
original PyTorch model before trusting it further (quantization, QNN).

VAE encoder note: exports the distribution MEAN, not a stochastic sample --
standard practice for deterministic autoencoder-style encoding (also removes
injected noise, which should slightly help reconstruction determinism/
quality). This is a real behavior change from the current pipeline.encode()
path (which samples); verify PSNR still holds before adopting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

import cv2

from imgint.codec import SdxlVaeCodec, TARGET_SIZE, resize_short_side_and_center_crop

# a previously-captured real photo -- export/verification doesn't need a live
# frame, and repeatedly grabbing the camera for this was hitting contention
# with the user's own camera use.
SAMPLE_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "static"
    / "captures"
    / "e2f8a41ab1ad4157875150f597be1283.jpg"
)

ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)


class SiglipImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values).pooler_output


class SiglipTextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.get_text_features(input_ids=input_ids).pooler_output


class VaeEncoderMean(torch.nn.Module):
    """Exports the deterministic mean of the latent distribution, not a
    stochastic sample -- see module docstring.
    """

    def __init__(self, vae, scaling_factor):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor

    def forward(self, pixel_values):
        latent_dist = self.vae.encode(pixel_values).latent_dist
        return latent_dist.mean * self.scaling_factor


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


if __name__ == "__main__":
    print(f"loading a real saved photo for export/verification inputs: {SAMPLE_IMAGE_PATH.name}")
    frame = cv2.imread(str(SAMPLE_IMAGE_PATH))  # BGR, matches what capture_frame() returns

    print("\n=== SigLIP image tower ===")
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").eval()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image_inputs = processor(images=image, return_tensors="pt")
    print("image processor output keys:", list(image_inputs.keys()))
    pixel_values = image_inputs["pixel_values"]

    with torch.no_grad():
        torch_image_embed = model.get_image_features(pixel_values=pixel_values).pooler_output.numpy()

    image_encoder = SiglipImageEncoder(model)
    image_onnx_path = ONNX_DIR / "siglip_image.onnx"
    torch.onnx.export(
        image_encoder,
        (pixel_values,),
        str(image_onnx_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {image_onnx_path} ({image_onnx_path.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(image_onnx_path), providers=["CPUExecutionProvider"])
    onnx_image_embed = sess.run(None, {"pixel_values": pixel_values.numpy()})[0]
    diff = max_abs_diff(torch_image_embed, onnx_image_embed)
    print(f"max abs diff (torch vs onnx, image tower): {diff:.6f}")
    assert diff < 1e-3, "ONNX image tower output diverges too much from PyTorch"

    print("\n=== SigLIP text tower ===")
    text_inputs = processor(text=["a photo of a room"], padding="max_length", return_tensors="pt")
    print("text processor output keys:", list(text_inputs.keys()))
    input_ids = text_inputs["input_ids"]

    with torch.no_grad():
        torch_text_embed = model.get_text_features(input_ids=input_ids).pooler_output.numpy()

    text_encoder = SiglipTextEncoder(model)
    text_onnx_path = ONNX_DIR / "siglip_text.onnx"
    torch.onnx.export(
        text_encoder,
        (input_ids,),
        str(text_onnx_path),
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {text_onnx_path} ({text_onnx_path.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(text_onnx_path), providers=["CPUExecutionProvider"])
    onnx_text_embed = sess.run(None, {"input_ids": input_ids.numpy()})[0]
    diff = max_abs_diff(torch_text_embed, onnx_text_embed)
    print(f"max abs diff (torch vs onnx, text tower): {diff:.6f}")
    assert diff < 1e-3, "ONNX text tower output diverges too much from PyTorch"

    print("\n=== VAE encoder (mean, not sample) ===")
    codec = SdxlVaeCodec()
    cropped = resize_short_side_and_center_crop(image, TARGET_SIZE)
    arr = (np.asarray(cropped).astype(np.float32) / 255.0) * 2 - 1
    vae_pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    vae_encoder = VaeEncoderMean(codec.vae, codec.vae.config.scaling_factor)
    with torch.no_grad():
        torch_latent_mean = vae_encoder(vae_pixel_values).numpy()

    vae_onnx_path = ONNX_DIR / "vae_encoder.onnx"
    torch.onnx.export(
        vae_encoder,
        (vae_pixel_values,),
        str(vae_onnx_path),
        input_names=["pixel_values"],
        output_names=["latent"],
        dynamic_axes={"pixel_values": {0: "batch"}, "latent": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {vae_onnx_path} ({vae_onnx_path.stat().st_size / 1e6:.1f} MB)")

    sess = ort.InferenceSession(str(vae_onnx_path), providers=["CPUExecutionProvider"])
    onnx_latent = sess.run(None, {"pixel_values": vae_pixel_values.numpy()})[0]
    diff = max_abs_diff(torch_latent_mean, onnx_latent)
    print(f"max abs diff (torch vs onnx, VAE encoder): {diff:.6f}")
    assert diff < 1e-3, "ONNX VAE encoder output diverges too much from PyTorch"

    print("\nAll three ONNX exports verified against PyTorch originals.")
