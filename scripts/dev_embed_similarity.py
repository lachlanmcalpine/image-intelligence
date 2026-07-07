"""Milestone 2 verification: embed a visually-similar pair and a clearly
different image, print pairwise cosine similarities, and assert the similar
pair scores higher. Uses synthetic images so this doesn't depend on the
webcam or any external file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from imgint.embedder import SigLipEmbedder, cosine_similarity


def make_test_images():
    # a / b: visually similar (same red circle, slightly different position/shade)
    img_a = Image.new("RGB", (256, 256), (235, 235, 235))
    ImageDraw.Draw(img_a).ellipse([60, 60, 190, 190], fill=(200, 30, 30))

    img_b = Image.new("RGB", (256, 256), (225, 225, 225))
    ImageDraw.Draw(img_b).ellipse([70, 65, 195, 195], fill=(190, 25, 25))

    # c: clearly different scene
    img_c = Image.new("RGB", (256, 256), (20, 20, 80))
    ImageDraw.Draw(img_c).rectangle([40, 40, 216, 216], fill=(230, 220, 40))

    return img_a, img_b, img_c


if __name__ == "__main__":
    print("loading SigLIP base (first run downloads ~813 MB)...")
    embedder = SigLipEmbedder()

    img_a, img_b, img_c = make_test_images()
    vec_a = embedder.embed(img_a)
    vec_b = embedder.embed(img_b)
    vec_c = embedder.embed(img_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)
    sim_bc = cosine_similarity(vec_b, vec_c)

    print(f"embedding dim: {vec_a.shape}")
    print(f"sim(similar pair a,b)    = {sim_ab:.4f}")
    print(f"sim(dissimilar pair a,c) = {sim_ac:.4f}")
    print(f"sim(dissimilar pair b,c) = {sim_bc:.4f}")

    assert sim_ab > sim_ac, "expected similar pair to score higher than a dissimilar pair"
    assert sim_ab > sim_bc, "expected similar pair to score higher than a dissimilar pair"
    print("PASS: similar pair scored higher than dissimilar pairs")
