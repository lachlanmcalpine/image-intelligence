import numpy as np

from imgint.embedder import cosine_similarity


def test_fake_embedder_deterministic(fake_embedder):
    vec1 = fake_embedder.embed("image-a")
    vec2 = fake_embedder.embed("image-a")
    assert np.allclose(vec1, vec2)


def test_fake_embedder_differs_for_different_input(fake_embedder):
    vec1 = fake_embedder.embed("image-a")
    vec2 = fake_embedder.embed("image-b")
    assert not np.allclose(vec1, vec2)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6
