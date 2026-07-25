"""Run the 6 images from the Phase 3b VAE-resolution comparison (original +
reconstructed, at 128/256/512) through Real-ESRGAN's general x4 restoration
model, to see whether AI enhancement recovers detail lost by low-resolution
VAE reconstruction (especially the blurrier 128^2 case).

Deliberately avoids the `realesrgan`/`basicsr` PyPI packages -- `basicsr`
fails to even build on this machine (its setup.py can't read its own
version file with modern setuptools), on top of the already-documented
torchvision.transforms.functional_tensor removal (see
requirements-realesrgan.txt). Instead this re-implements the small,
MIT-licensed SRVGGNetCompact architecture directly in pure PyTorch and loads
the official realesr-general-x4v3.pth checkpoint -- no basicsr needed for
inference, only for training utilities we don't use.

face_enhance is intentionally not offered -- it invokes GFPGAN, which
hallucinates face detail, contradicting the "faithful reconstruction, no
hallucination" goal for this project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "models" / "realesr-general-x4v3.pth"
IN_DIR = REPO_ROOT / "out" / "vae_res"
OUT_DIR = REPO_ROOT / "out" / "vae_res_enhanced"

IMAGES = [
    "original_128.jpg",
    "reconstructed_128.jpg",
    "original_256.jpg",
    "reconstructed_256.jpg",
    "original_512.jpg",
    "reconstructed_512.jpg",
]


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style super-resolution network -- the architecture behind
    Real-ESRGAN's 'general' x4 models. Conv+activation stack followed by a
    pixel-shuffle upsampler, with a nearest-neighbor-upscaled skip connection
    added back at the end. Matches realesr-general-x4v3.pth exactly:
    num_feat=64, num_conv=32, upscale=4, prelu activations.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def load_model() -> SRVGGNetCompact:
    model = SRVGGNetCompact()
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    key = "params_ema" if "params_ema" in state else "params"
    model.load_state_dict(state[key], strict=True)
    model.eval()
    return model


def enhance(model: SRVGGNetCompact, bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    out = out.clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    if not WEIGHTS_PATH.exists():
        print(f"missing weights: {WEIGHTS_PATH}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading realesr-general-x4v3 (SRVGGNetCompact, pure PyTorch, no basicsr)...")
    model = load_model()

    for name in IMAGES:
        in_path = IN_DIR / name
        if not in_path.exists():
            print(f"skipping missing {in_path}")
            continue
        bgr = cv2.imread(str(in_path))
        h, w = bgr.shape[:2]

        enhanced = enhance(model, bgr)
        out_path = OUT_DIR / name
        cv2.imwrite(str(out_path), enhanced)
        eh, ew = enhanced.shape[:2]
        print(f"{name}: {w}x{h} -> {ew}x{eh}  saved -> {out_path.relative_to(REPO_ROOT)}")

    print("\ndone.")
