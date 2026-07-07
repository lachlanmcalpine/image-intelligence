import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from imgint.crypto import (
    decrypt_latent,
    encrypt_latent,
    generate_aes_key,
    generate_orthogonal_matrix,
    transform_embedding,
)
from imgint.embedder import cosine_similarity


def test_orthogonal_matrix_is_actually_orthogonal():
    rng = np.random.default_rng(0)
    m = generate_orthogonal_matrix(16, rng=rng)
    identity = m @ m.T
    assert np.allclose(identity, np.eye(16), atol=1e-5)


def test_transform_preserves_cosine_similarity():
    rng = np.random.default_rng(1)
    m = generate_orthogonal_matrix(32, rng=rng)

    a = rng.normal(size=32).astype(np.float32)
    b = rng.normal(size=32).astype(np.float32)

    sim_before = cosine_similarity(a, b)
    sim_after = cosine_similarity(transform_embedding(a, m), transform_embedding(b, m))

    assert abs(sim_before - sim_after) < 1e-4


def test_different_users_get_different_matrices():
    m1 = generate_orthogonal_matrix(16, rng=np.random.default_rng(1))
    m2 = generate_orthogonal_matrix(16, rng=np.random.default_rng(2))
    assert not np.allclose(m1, m2)


def test_aes_gcm_round_trip_recovers_original_bytes():
    key = generate_aes_key()
    original = b"a fake VAE latent, as raw bytes"

    nonce, ciphertext = encrypt_latent(original, key)
    recovered = decrypt_latent(nonce, ciphertext, key)

    assert recovered == original


def test_aes_gcm_rejects_tampered_ciphertext():
    key = generate_aes_key()
    nonce, ciphertext = encrypt_latent(b"some latent bytes", key)

    tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]

    with pytest.raises(InvalidTag):
        decrypt_latent(nonce, tampered, key)


def test_aes_gcm_rejects_wrong_key():
    key_a = generate_aes_key()
    key_b = generate_aes_key()
    nonce, ciphertext = encrypt_latent(b"some latent bytes", key_a)

    with pytest.raises(InvalidTag):
        decrypt_latent(nonce, ciphertext, key_b)
