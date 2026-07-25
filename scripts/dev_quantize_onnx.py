"""Static-quantize the exported ONNX models to INT8 (QDQ format), which QNN's
NPU backend needs -- dynamic quantization (scale computed per-inference) is
not well supported by fixed-point NPU hardware; the scale/zero-point must be
calibrated ahead of time from representative data.

Calibration set is small (only 2 raw real captures exist on disk this
session) -- augmented with brightness/contrast/crop variants for a bit more
coverage. A real deployment would want more diverse samples; flag this if
quantized quality looks off.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image, ImageEnhance
from transformers import SiglipProcessor

from imgint.codec import TARGET_SIZE, resize_short_side_and_center_crop

ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"

RAW_CAPTURES = [
    ONNX_DIR.parent.parent / "static" / "captures" / "e2f8a41ab1ad4157875150f597be1283.jpg",
    ONNX_DIR.parent.parent / "out" / "frame_0001.jpg",
]

SAMPLE_QUESTIONS = [
    "did I see my wallet today?",
    "what room is this?",
    "describe what you see in this image",
    "is there a person in the photo?",
    "what color are the walls?",
    "was the door open or closed?",
]


def load_augmented_images() -> list:
    images = []
    for path in RAW_CAPTURES:
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        images.append(pil_img)
        # a few cheap augmentations for calibration variety
        images.append(ImageEnhance.Brightness(pil_img).enhance(1.3))
        images.append(ImageEnhance.Brightness(pil_img).enhance(0.7))
        images.append(ImageEnhance.Contrast(pil_img).enhance(1.3))
    return images


class ImageTowerCalibrationReader(CalibrationDataReader):
    def __init__(self, images):
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.data = iter(
            [{"pixel_values": processor(images=img, return_tensors="np")["pixel_values"]} for img in images]
        )

    def get_next(self):
        return next(self.data, None)


class VaeCalibrationReader(CalibrationDataReader):
    def __init__(self, images):
        prepared = []
        for img in images:
            cropped = resize_short_side_and_center_crop(img, TARGET_SIZE)
            arr = (np.asarray(cropped).astype(np.float32) / 255.0) * 2 - 1
            arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
            prepared.append({"pixel_values": arr})
        self.data = iter(prepared)

    def get_next(self):
        return next(self.data, None)


class TextTowerCalibrationReader(CalibrationDataReader):
    def __init__(self, questions):
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.data = iter(
            [
                {"input_ids": processor(text=[q], padding="max_length", return_tensors="np")["input_ids"]}
                for q in questions
            ]
        )

    def get_next(self):
        return next(self.data, None)


def quantize(model_name: str, reader, extra_options: dict | None = None) -> None:
    input_path = ONNX_DIR / f"{model_name}.onnx"
    output_path = ONNX_DIR / f"{model_name}_int8.onnx"
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        extra_options=extra_options or {},
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"{model_name}: {input_path.stat().st_size/1e6:.1f} MB -> {output_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("loading calibration images...")
    images = load_augmented_images()
    print(f"calibration set: {len(images)} image variants, {len(SAMPLE_QUESTIONS)} text queries")

    print("\nquantizing SigLIP image tower...")
    quantize("siglip_image", ImageTowerCalibrationReader(images))

    print("\nquantizing SigLIP text tower...")
    quantize("siglip_text", TextTowerCalibrationReader(SAMPLE_QUESTIONS))

    print("\nquantizing VAE encoder...")
    quantize("vae_encoder", VaeCalibrationReader(images))

    print("\ndone.")
