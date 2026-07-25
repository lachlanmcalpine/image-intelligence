"""Verify the quantized (INT8) ONNX models still work correctly, running on
plain CPUExecutionProvider first (before any QNN-specific concerns): SigLIP
similarity ordering preserved, VAE reconstruction quality still acceptable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from transformers import SiglipProcessor

from imgint.codec import SdxlVaeCodec, TARGET_SIZE, resize_short_side_and_center_crop

ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"
SAMPLE_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent / "static" / "captures" / "e2f8a41ab1ad4157875150f597be1283.jpg"
)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def make_test_images():
    img_a = Image.new("RGB", (256, 256), (235, 235, 235))
    ImageDraw.Draw(img_a).ellipse([60, 60, 190, 190], fill=(200, 30, 30))
    img_b = Image.new("RGB", (256, 256), (225, 225, 225))
    ImageDraw.Draw(img_b).ellipse([70, 65, 195, 195], fill=(190, 25, 25))
    img_c = Image.new("RGB", (256, 256), (20, 20, 80))
    ImageDraw.Draw(img_c).rectangle([40, 40, 216, 216], fill=(230, 220, 40))
    return img_a, img_b, img_c


if __name__ == "__main__":
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

    print("=== SigLIP image tower (int8) similarity check ===")
    sess = ort.InferenceSession(
        str(ONNX_DIR / "siglip_image_int8.onnx"), providers=["CPUExecutionProvider"]
    )
    img_a, img_b, img_c = make_test_images()
    embeds = {}
    for name, img in [("a", img_a), ("b", img_b), ("c", img_c)]:
        pixel_values = processor(images=img, return_tensors="np")["pixel_values"]
        embeds[name] = sess.run(None, {"pixel_values": pixel_values})[0][0]

    sim_ab = cosine_similarity(embeds["a"], embeds["b"])
    sim_ac = cosine_similarity(embeds["a"], embeds["c"])
    print(f"sim(similar pair a,b)    = {sim_ab:.4f}")
    print(f"sim(dissimilar pair a,c) = {sim_ac:.4f}")
    assert sim_ab > sim_ac, "int8 image tower broke similarity ordering!"
    print("PASS: similarity ordering preserved\n")

    print("=== SigLIP text tower (int8) cross-modal check ===")
    text_sess = ort.InferenceSession(
        str(ONNX_DIR / "siglip_text_int8.onnx"), providers=["CPUExecutionProvider"]
    )
    input_ids = processor(text=["a red circle"], padding="max_length", return_tensors="np")["input_ids"]
    text_embed = text_sess.run(None, {"input_ids": input_ids})[0][0]
    sim_text_a = cosine_similarity(text_embed, embeds["a"])
    sim_text_c = cosine_similarity(text_embed, embeds["c"])
    print(f"sim('a red circle' vs red-circle image) = {sim_text_a:.4f}")
    print(f"sim('a red circle' vs yellow-square image) = {sim_text_c:.4f}")
    assert sim_text_a > sim_text_c, "int8 text tower broke cross-modal ordering!"
    print("PASS: cross-modal similarity ordering preserved\n")

    print("=== VAE encoder (int8) reconstruction quality ===")
    frame = cv2.imread(str(SAMPLE_IMAGE_PATH))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    cropped = resize_short_side_and_center_crop(image, TARGET_SIZE)
    real_photo_256 = np.asarray(cropped)
    arr = (real_photo_256.astype(np.float32) / 255.0) * 2 - 1
    pixel_values = arr.transpose(2, 0, 1)[np.newaxis, ...]

    vae_sess = ort.InferenceSession(
        str(ONNX_DIR / "vae_encoder_int8.onnx"), providers=["CPUExecutionProvider"]
    )
    latent = vae_sess.run(None, {"pixel_values": pixel_values})[0][0]  # already scaled

    codec = SdxlVaeCodec()
    latent_tensor = torch.from_numpy(latent).unsqueeze(0)
    with torch.no_grad():
        decoded = codec.vae.decode(latent_tensor / codec.vae.config.scaling_factor).sample[0]
    decoded = (decoded.clamp(-1, 1) + 1) / 2 * 255
    reconstructed = decoded.permute(1, 2, 0).numpy().astype(np.uint8)

    score = psnr(real_photo_256, reconstructed)
    print(f"PSNR (int8 VAE encoder vs real photo): {score:.2f} dB")
    print("(for reference: fp32 baseline was ~32 dB, our MVP quality bar was >26 dB)")

    out_path = Path(__file__).resolve().parent.parent / "out" / "int8_onnx_vae_check.jpg"
    side_by_side = np.concatenate([real_photo_256, reconstructed], axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    print(f"saved original|reconstructed -> {out_path}")
