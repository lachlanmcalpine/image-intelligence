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


def test_ranks_by_cosine_similarity_not_l2_distance(store):
    """Regression test: Chroma defaults to L2 distance unless hnsw:space is
    explicitly set to cosine. transform_embedding() (imgint/crypto.py) only
    preserves *cosine* similarity under the orthogonal transform, and SigLIP
    embeddings aren't unit-normalized -- so an L2-space collection ranks
    nearest neighbors differently than intended and silently returns the
    wrong match. This is exactly what happened live: distances in the
    hundreds and the wrong photo retrieved.

    Constructed so L2 and cosine disagree on the nearest neighbor: `close_in_angle`
    points in nearly the same direction as the query (high cosine similarity)
    but is far away in raw distance (large magnitude); `close_in_distance` is
    numerically near the query but points in a clearly different direction.
    """
    query = np.array([1.0, 0.0], dtype=np.float32)
    close_in_angle = np.array([10.0, 0.0], dtype=np.float32)  # cos=1.0, L2 distance=9.0
    close_in_distance = np.array([1.5, 1.5], dtype=np.float32)  # cos=0.707, L2 distance=1.58

    store.upsert("close_in_angle", close_in_angle, nonce=b"n" * 12, ciphertext=b"c" * 16)
    store.upsert("close_in_distance", close_in_distance, nonce=b"n" * 12, ciphertext=b"c" * 16)

    matches = store.query(query, top_k=1)

    assert matches[0]["id"] == "close_in_angle"


def test_query_where_filters_by_timestamp(store):
    """Time must scope the search itself, not post-filter its results --
    otherwise "what did I do this morning" returns whatever is globally most
    similar and then drops it, instead of the best match *within* the window.
    """
    rng = np.random.default_rng(3)
    vec = rng.normal(size=16).astype(np.float32)
    for ts, id_ in [(100.0, "old"), (200.0, "mid"), (300.0, "new")]:
        store.upsert(
            id_,
            vec + rng.normal(scale=0.01, size=16).astype(np.float32),
            nonce=b"n" * 12,
            ciphertext=b"c" * 16,
            extra_metadata={"timestamp": ts},
        )

    where = {"$and": [{"timestamp": {"$gte": 150.0}}, {"timestamp": {"$lte": 250.0}}]}
    matches = store.query(vec, top_k=5, where=where)

    assert [m["id"] for m in matches] == ["mid"]

    # unscoped query still sees everything
    assert len(store.query(vec, top_k=5)) == 3


def test_get_by_text_substring_search_case_insensitive(store):
    rng = np.random.default_rng(9)
    store.upsert("a", rng.normal(size=8).astype(np.float32), b"n" * 12, b"c" * 16,
                 extra_metadata={"timestamp": 1.0}, document="INVOICE number 88 total 42")
    store.upsert("b", rng.normal(size=8).astype(np.float32), b"n" * 12, b"c" * 16,
                 extra_metadata={"timestamp": 2.0}, document="a photo of a wall, no text")

    # matches regardless of case (the trap that caused live "nothing found")
    for query in ("INVOICE", "invoice", "Invoice", "voice"):
        hits = store.get_by_text(query)
        assert [h["id"] for h in hits] == ["a"], query
    assert store.get_by_text("nothing-like-this") == []


def test_get_recent_newest_first(store):
    rng = np.random.default_rng(20)
    for id_, ts in [("old", 100.0), ("new", 300.0), ("mid", 200.0)]:
        store.upsert(id_, rng.normal(size=8).astype(np.float32), b"n" * 12, b"c" * 16,
                     extra_metadata={"timestamp": ts}, document=f"doc {id_}")

    recent = store.get_recent(limit=2)
    assert [r["id"] for r in recent] == ["new", "mid"]
    assert recent[0]["document"] == "doc new"


def test_get_by_text_orders_newest_first_and_scopes_by_time(store):
    rng = np.random.default_rng(10)
    for id_, ts in [("old", 100.0), ("new", 300.0), ("mid", 200.0)]:
        store.upsert(id_, rng.normal(size=8).astype(np.float32), b"n" * 12, b"c" * 16,
                     extra_metadata={"timestamp": ts}, document="receipt")

    assert [h["id"] for h in store.get_by_text("receipt")] == ["new", "mid", "old"]

    scoped = store.get_by_text("receipt", where={"timestamp": {"$gte": 250.0}})
    assert [h["id"] for h in scoped] == ["new"]


def test_upsert_without_document_is_not_text_searchable(store):
    rng = np.random.default_rng(11)
    store.upsert("x", rng.normal(size=8).astype(np.float32), b"n" * 12, b"c" * 16,
                 extra_metadata={"timestamp": 1.0})  # no document
    # empty/absent documents must not crash text search or match everything
    assert store.get_by_text("anything") == []


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
