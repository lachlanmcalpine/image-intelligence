import ast
import uuid
from pathlib import Path

import numpy as np
import pytest

from imgint.store import Store, make_client


@pytest.fixture
def store():
    # chromadb's EphemeralClient shares underlying state across instantiations
    # within the same process (collections of the same name collide across
    # tests), so give each test its own uniquely-named collection.
    client = make_client("ephemeral")
    return Store(client, collection_name=f"test-{uuid.uuid4().hex}")


def test_upsert_and_query_returns_nearest_match(store):
    rng = np.random.default_rng(0)
    vec_a = rng.normal(size=32).astype(np.float32)
    vec_b = rng.normal(size=32).astype(np.float32)
    vec_c = rng.normal(size=32).astype(np.float32)

    store.upsert("a", vec_a, nonce=b"nonce-a-000\x00", ciphertext=b"ciphertext-for-a")
    store.upsert("b", vec_b, nonce=b"nonce-b-000\x00", ciphertext=b"ciphertext-for-b")
    store.upsert("c", vec_c, nonce=b"nonce-c-000\x00", ciphertext=b"ciphertext-for-c")

    # query with something very close to vec_a
    query_vec = vec_a + rng.normal(scale=0.001, size=32).astype(np.float32)
    matches = store.query(query_vec, top_k=1)

    assert len(matches) == 1
    assert matches[0]["id"] == "a"
    assert matches[0]["nonce"] == b"nonce-a-000\x00"
    assert matches[0]["ciphertext"] == b"ciphertext-for-a"


def test_query_top_k_returns_requested_count(store):
    rng = np.random.default_rng(1)
    for i in range(5):
        vec = rng.normal(size=16).astype(np.float32)
        store.upsert(f"frame-{i}", vec, nonce=b"n" * 12, ciphertext=b"c" * 16)

    matches = store.query(rng.normal(size=16).astype(np.float32), top_k=3)
    assert len(matches) == 3


def test_upsert_preserves_extra_metadata(store):
    rng = np.random.default_rng(2)
    vec = rng.normal(size=8).astype(np.float32)
    store.upsert(
        "frame-x",
        vec,
        nonce=b"n" * 12,
        ciphertext=b"c" * 16,
        extra_metadata={"source": "webcam"},
    )
    matches = store.query(vec, top_k=1)
    assert matches[0]["metadata"]["source"] == "webcam"


def test_store_module_never_imports_crypto():
    """The entire confidentiality boundary depends on M and the AES key never
    leaving this device -- structurally enforce that store.py (which talks to
    the cloud) doesn't even import the crypto module.
    """
    store_path = Path(__file__).resolve().parent.parent / "imgint" / "store.py"
    tree = ast.parse(store_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any("crypto" in m for m in imported_modules), (
        f"store.py must never import imgint.crypto -- found imports: {imported_modules}"
    )
