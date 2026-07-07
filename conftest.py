"""Shared pytest fixtures. Fakes let logic tests run fast without loading any
real model, camera, or network — mirrors face-anonymiser's FakeBackend pattern.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic stand-in for SigLipEmbedder: same input -> same vector,
    different input -> different vector, no model/torch involved.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, image) -> np.ndarray:
        seed = abs(hash(repr(image))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dim).astype(np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
