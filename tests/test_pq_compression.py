import numpy as np
import pytest

from imgint.embedder import cosine_similarity
from imgint.pq_compression import decode_pq, encode_pq, train_pq_codebooks

DIM = 64
NUM_SUBVECTORS = 8  # 8-dim subspaces
NUM_CENTROIDS = 16


CLUSTER_CENTERS = np.random.default_rng(0).normal(scale=5.0, size=(5, DIM)).astype(np.float32)


def _sample_from_centers(centers=CLUSTER_CENTERS, points_per_cluster=40, noise_scale=0.2, seed=0):
    """Synthetic stand-in for real SigLIP embeddings: a few fixed,
    well-separated cluster centers (like distinct scenes) plus small
    per-point noise (like near-duplicate frames of the same scene). Centers
    are shared across calls -- only the noise varies by seed -- so a "fresh
    sample" is actually a fresh sample of the *same* clusters, not new ones.
    """
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    for i, center in enumerate(centers):
        noise = rng.normal(scale=noise_scale, size=(points_per_cluster, len(center))).astype(np.float32)
        points.append(center + noise)
        labels.extend([i] * points_per_cluster)
    return np.concatenate(points, axis=0), np.array(labels)


@pytest.fixture
def codebooks():
    vectors, _labels = _sample_from_centers(seed=1)
    return train_pq_codebooks(vectors, num_subvectors=NUM_SUBVECTORS, num_centroids=NUM_CENTROIDS, seed=1)


def test_compression_ratio(codebooks):
    vector = np.zeros(DIM, dtype=np.float32)
    codes = encode_pq(vector, codebooks)

    raw_bytes = vector.astype(np.float32).nbytes
    compressed_bytes = codes.nbytes

    assert compressed_bytes == NUM_SUBVECTORS  # one byte per subvector
    assert raw_bytes / compressed_bytes >= 30  # DIM=64 gives ~32x here; 768-dim gives ~384x


def test_round_trip_shape_and_dtype(codebooks):
    vector = np.random.default_rng(2).normal(size=DIM).astype(np.float32)
    codes = encode_pq(vector, codebooks)
    reconstructed = decode_pq(codes, codebooks)

    assert codes.dtype == np.uint8
    assert codes.shape == (NUM_SUBVECTORS,)
    assert reconstructed.shape == (DIM,)
    assert reconstructed.dtype == np.float32


def test_reconstruction_is_close_for_in_distribution_vectors(codebooks):
    vectors, _labels = _sample_from_centers(seed=99)  # fresh noise sample, same fixed clusters
    similarities = []
    for v in vectors:
        codes = encode_pq(v, codebooks)
        reconstructed = decode_pq(codes, codebooks)
        similarities.append(cosine_similarity(v, reconstructed))

    # lossy, but for tight, well-separated clusters the nearest centroid
    # should still be a good stand-in for the real vector
    assert min(similarities) > 0.9


def test_ranking_is_preserved_against_true_cosine_similarity(codebooks):
    """The property that actually matters for retrieval: does the PQ-lossy
    ranking agree with the ranking you'd get from the real, unquantized
    vectors? This is the same category of bug as the Chroma L2-vs-cosine
    issue -- a compression scheme can round-trip fine in isolation while
    still silently reordering search results.
    """
    rng = np.random.default_rng(7)

    query = CLUSTER_CENTERS[0] + rng.normal(scale=0.2, size=DIM).astype(np.float32)
    candidates = [
        CLUSTER_CENTERS[i] + rng.normal(scale=0.2, size=DIM).astype(np.float32)
        for i in range(len(CLUSTER_CENTERS))
    ]

    true_similarities = [cosine_similarity(query, c) for c in candidates]
    true_best = int(np.argmax(true_similarities))

    pq_similarities = []
    for c in candidates:
        codes = encode_pq(c, codebooks)
        reconstructed = decode_pq(codes, codebooks)
        pq_similarities.append(cosine_similarity(query, reconstructed))
    pq_best = int(np.argmax(pq_similarities))

    assert pq_best == true_best == 0  # candidates[0] is drawn from the same cluster as the query


def test_rejects_dim_not_divisible_by_num_subvectors():
    vectors = np.zeros((100, 65), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        train_pq_codebooks(vectors, num_subvectors=8, num_centroids=16)


def test_rejects_too_few_training_vectors():
    vectors = np.zeros((10, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="training vectors"):
        train_pq_codebooks(vectors, num_subvectors=NUM_SUBVECTORS, num_centroids=16)


def test_rejects_too_many_centroids_for_uint8():
    vectors = np.zeros((1000, DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="256"):
        train_pq_codebooks(vectors, num_subvectors=NUM_SUBVECTORS, num_centroids=257)
