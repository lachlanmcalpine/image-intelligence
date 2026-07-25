"""Product quantization (PQ) for SigLIP embeddings.

A 768-dim fp32 embedding (3,072 bytes) is split into `num_subvectors` equal
chunks; a k-means codebook is trained per chunk (offline, on a representative
sample of real embeddings), and each chunk is then replaced by the index of
its nearest centroid -- one byte per chunk (assuming <=256 centroids). An
8-subvector split stores a whole embedding in 8 bytes: ~384x smaller than raw
fp32, at the cost of a lossy, approximate reconstruction.

This only touches the embedding (used for search ranking) -- it is a
DIFFERENT lever from the VAE-latent int8 quantization in compression.py,
which shrinks the *image* payload. The two compose; they don't overlap.

Deliberately implemented with plain numpy (Lloyd's-algorithm k-means) instead
of adding scikit-learn as a dependency -- this project keeps its dependency
list intentionally small, and k-means over a handful of 96-dim subspaces is
simple enough not to need a library.

IMPORTANT (not yet wired into imgint/store.py): Chroma's own HNSW index
expects real float vectors to compute distances on -- it has no notion of
"these bytes are PQ codes, decode before comparing." Storing raw PQ codes as
a Chroma embedding would make its ANN search meaningless. Using this for live
retrieval requires an index that understands PQ natively (e.g. FAISS's
IndexIVFPQ, or Milvus/Weaviate/Qdrant's quantization support) -- a store
migration, not a drop-in change. See todo.md.
"""

import numpy as np

MAX_CENTROIDS = 256  # one byte per subvector code


def train_pq_codebooks(
    vectors: np.ndarray,
    num_subvectors: int = 8,
    num_centroids: int = 256,
    num_iters: int = 25,
    seed: int | None = None,
) -> np.ndarray:
    """Trains one k-means codebook per subvector on a representative sample
    of real embeddings. Returns codebooks, shape (num_subvectors, num_centroids,
    dim // num_subvectors).

    Needs at least num_centroids training vectors (k-means can't produce more
    clusters than it has points) -- in practice, use a few thousand real
    embeddings, not a handful.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    if dim % num_subvectors != 0:
        raise ValueError(f"embedding dim {dim} not divisible by num_subvectors {num_subvectors}")
    if num_centroids > MAX_CENTROIDS:
        raise ValueError(f"num_centroids must be <= {MAX_CENTROIDS} to fit in a uint8 code")
    if n < num_centroids:
        raise ValueError(f"need at least {num_centroids} training vectors, got {n}")

    sub_dim = dim // num_subvectors
    rng = np.random.default_rng(seed)
    codebooks = np.empty((num_subvectors, num_centroids, sub_dim), dtype=np.float32)
    for m in range(num_subvectors):
        sub = vectors[:, m * sub_dim : (m + 1) * sub_dim]
        codebooks[m] = _kmeans(sub, num_centroids, num_iters, rng)
    return codebooks


def _kmeans(data: np.ndarray, k: int, num_iters: int, rng: np.random.Generator) -> np.ndarray:
    """Lloyd's algorithm, plain numpy. Centroids initialized from random
    distinct data points (Forgy init) -- fine at this scale; not worth
    k-means++ complexity for a few hundred training vectors per subspace.
    """
    n = data.shape[0]
    centroids = data[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(num_iters):
        dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = dists.argmin(axis=1)
        for c in range(k):
            members = data[assignments == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return centroids


def encode_pq(vector: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Returns a (num_subvectors,) uint8 array: the nearest-centroid index in
    each subspace. This is the compressed, storable form.
    """
    num_subvectors, _num_centroids, sub_dim = codebooks.shape
    codes = np.empty(num_subvectors, dtype=np.uint8)
    for m in range(num_subvectors):
        sub = vector[m * sub_dim : (m + 1) * sub_dim]
        dists = ((codebooks[m] - sub) ** 2).sum(axis=1)
        codes[m] = dists.argmin()
    return codes


def decode_pq(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Reconstructs an approximate (dim,) float32 vector from PQ codes by
    concatenating each subspace's assigned centroid. Lossy -- this recovers
    "the nearest centroid," not the original vector.
    """
    num_subvectors, _num_centroids, _sub_dim = codebooks.shape
    parts = [codebooks[m, codes[m]] for m in range(num_subvectors)]
    return np.concatenate(parts).astype(np.float32)
