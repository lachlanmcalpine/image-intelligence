"""Client-side encryption: a per-user orthogonal-matrix transform on
embeddings (preserves cosine similarity for search while obscuring the raw
vector) and AES-GCM on VAE latents (image content). Both keys are local-only
-- gitignored under ./keys/, never touch store.py's upsert payload or the
DeepSeek call.

Known limitation (accepted for MVP, see todo.md): the orthogonal-matrix
transform is obfuscation, not cryptographically hard encryption -- enough
known (embedding, transformed_embedding) pairs could let an attacker recover
`M` via least-squares/Procrustes. Per-user matrices limit the blast radius of
a break to a single user.
"""

import os
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEYS_DIR = Path(__file__).resolve().parent.parent / "keys"
NONCE_SIZE = 12  # AES-GCM standard nonce size, in bytes


def generate_orthogonal_matrix(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """A uniform-random orthogonal matrix via QR decomposition of a Gaussian
    matrix, with the QR sign ambiguity fixed so the distribution is uniform
    over O(dim) rather than biased towards the identity.
    """
    rng = rng or np.random.default_rng()
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    d[d == 0] = 1
    return (q * d).astype(np.float32)


def transform_embedding(vec: np.ndarray, m: np.ndarray) -> np.ndarray:
    """cos(a @ M, b @ M) == cos(a, b) for orthogonal M -- ANN search on the
    transformed vectors returns the same neighbors as on the raw ones.
    """
    return vec @ m


def save_matrix(m: np.ndarray, path: Path | None = None) -> Path:
    path = path or (KEYS_DIR / "M.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, m)
    return path


def load_matrix(path: Path | None = None) -> np.ndarray:
    path = path or (KEYS_DIR / "M.npy")
    return np.load(path)


def generate_aes_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def save_aes_key(key: bytes, path: Path | None = None) -> Path:
    path = path or (KEYS_DIR / "aes.key")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    return path


def load_aes_key(path: Path | None = None) -> bytes:
    path = path or (KEYS_DIR / "aes.key")
    return path.read_bytes()


def ensure_keys(dim: int) -> tuple[np.ndarray, bytes]:
    """Load the per-user matrix + AES key from ./keys/, generating and
    persisting them on first use.
    """
    matrix_path = KEYS_DIR / "M.npy"
    key_path = KEYS_DIR / "aes.key"

    if matrix_path.exists():
        m = load_matrix(matrix_path)
    else:
        m = generate_orthogonal_matrix(dim)
        save_matrix(m, matrix_path)

    if key_path.exists():
        key = load_aes_key(key_path)
    else:
        key = generate_aes_key()
        save_aes_key(key, key_path)

    return m, key


def encrypt_latent(latent_bytes: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Returns (nonce, ciphertext). Ciphertext includes the GCM auth tag."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, latent_bytes, associated_data=None)
    return nonce, ciphertext


def decrypt_latent(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    """Raises cryptography.exceptions.InvalidTag if the ciphertext was
    tampered with or the wrong key/nonce is used -- fail closed, not silent.
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
