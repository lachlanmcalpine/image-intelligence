import time
import uuid

import numpy as np
import pytest

from imgint.crypto import generate_aes_key, generate_orthogonal_matrix
from imgint.pipeline import Pipeline
from imgint.store import Store, make_client

KEYS_DIM = 8  # matches fake_embedder's default dim


class _FakeCodec:
    def encode(self, frame) -> np.ndarray:
        seed = abs(hash(repr(frame))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=(4, 4, 4)).astype(np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        # deterministic pseudo-image derived from the latent, just needs to be
        # a valid ndarray for the fake answerer to receive
        value = int(abs(latent.sum() * 1000)) % 256
        return np.full((32, 32, 3), value, dtype=np.uint8)


class _FakeOcr:
    """Returns a fixed text per frame content so tests can assert what got
    stored and searched without a real OCR engine.
    """

    def __init__(self, text_by_value: dict[int, str] | None = None):
        self.text_by_value = text_by_value or {}
        self.calls = []

    def read_text(self, frame) -> str:
        self.calls.append(frame)
        return self.text_by_value.get(int(frame.flat[0]), "")


class _FakeAnswerer:
    def __init__(self):
        self.calls = []
        self.ask_many_calls = []

    def ask(self, image, question) -> str:
        self.calls.append((image, question))
        return f"fake answer to: {question}"

    def ask_many(self, images, question, labels=None) -> str:
        self.ask_many_calls.append((images, question, labels))
        return f"fake synthesis of {len(images)} images for: {question}"


@pytest.fixture
def pipeline(fake_embedder):
    client = make_client("ephemeral")
    store = Store(client, collection_name=f"test-pipeline-{uuid.uuid4().hex}")
    rng = np.random.default_rng(0)
    m = generate_orthogonal_matrix(KEYS_DIM, rng=rng)
    aes_key = generate_aes_key()
    return Pipeline(
        embedder=fake_embedder,
        codec=_FakeCodec(),
        store=store,
        answerer=_FakeAnswerer(),
        keys_dim=KEYS_DIM,
        m=m,
        aes_key=aes_key,
    )


def test_ingest_returns_a_record_id(pipeline):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    record_id = pipeline.ingest(frame)
    assert isinstance(record_id, str) and record_id


def test_query_after_ingest_returns_answer_and_image(pipeline):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pipeline.ingest(frame)

    results = pipeline.query("what is this?", top_k=1)

    assert len(results) == 1
    result = results[0]
    assert result["answer"] == "fake answer to: what is this?"
    assert isinstance(result["image"], np.ndarray)
    assert result["image"].shape == (32, 32, 3)


def test_query_calls_answerer_with_question_and_decoded_image(pipeline):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pipeline.ingest(frame)

    pipeline.query("is there a wallet?", top_k=1)

    assert len(pipeline.answerer.calls) == 1
    image, question = pipeline.answerer.calls[0]
    assert question == "is there a wallet?"
    assert isinstance(image, np.ndarray)


def test_query_top_k_returns_multiple_results(pipeline):
    for i in range(3):
        frame = np.full((64, 64, 3), i, dtype=np.uint8)
        pipeline.ingest(frame)

    results = pipeline.query("anything?", top_k=3)
    assert len(results) == 3


class _CountingEmbedder:
    """Wraps the fake embedder to count embed() calls -- lets us prove the
    pHash gate skips a frame *before* paying for an embedding.
    """

    def __init__(self, inner):
        self.inner = inner
        self.dim = inner.dim
        self.embed_calls = 0

    def embed(self, image):
        self.embed_calls += 1
        return self.inner.embed(image)

    def embed_text(self, text):
        return self.inner.embed_text(text)


def _make_pipeline(embedder, dedup_threshold=None, phash_threshold=None, ocr=None):
    client = make_client("ephemeral")
    store = Store(client, collection_name=f"test-pipeline-{uuid.uuid4().hex}")
    rng = np.random.default_rng(0)
    m = generate_orthogonal_matrix(KEYS_DIM, rng=rng)
    return Pipeline(
        embedder=embedder,
        codec=_FakeCodec(),
        store=store,
        answerer=_FakeAnswerer(),
        keys_dim=KEYS_DIM,
        m=m,
        aes_key=generate_aes_key(),
        dedup_threshold=dedup_threshold,
        phash_threshold=phash_threshold,
        ocr=ocr,
    )


def _structured_frame(seed: int) -> np.ndarray:
    """Per-seed structure via a random 8x8 block pattern upscaled -- gives
    dHash real spatial variation to distinguish, unlike a flat or gradient
    fill which would collide across seeds.
    """
    import cv2

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    gray = cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST)
    return np.stack([gray, gray, gray], axis=2)


def test_phash_skips_identical_frame_before_embedding(fake_embedder):
    embedder = _CountingEmbedder(fake_embedder)
    pipeline = _make_pipeline(embedder, phash_threshold=4)
    frame = _structured_frame(0)

    first = pipeline.ingest(frame)
    second = pipeline.ingest(frame.copy())  # identical pixels -> pHash skip

    assert first is not None
    assert second is None
    # the skipped frame never reached the embedder -- pHash gated it first
    assert embedder.embed_calls == 1


def test_phash_keeps_different_frames(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, phash_threshold=4)
    a = pipeline.ingest(_structured_frame(1))
    b = pipeline.ingest(_structured_frame(2))
    assert a is not None and b is not None


def test_phash_force_keeps_identical_frame(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, phash_threshold=4)
    frame = _structured_frame(3)
    pipeline.ingest(frame)
    assert pipeline.ingest(frame.copy(), force=True) is not None


def test_dedup_skips_identical_consecutive_frames(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, dedup_threshold=0.95)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    first = pipeline.ingest(frame)
    second = pipeline.ingest(frame)  # identical -> cosine sim 1.0 -> skipped

    assert first is not None
    assert second is None
    # only one record actually landed in the store
    assert len(pipeline.query("anything?", top_k=5)) == 1


def test_dedup_keeps_different_frames(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, dedup_threshold=0.95)
    # FakeEmbedder hashes frame content -> different frames get independent
    # random vectors, which are near-orthogonal (sim ~0) at any dim
    a = pipeline.ingest(np.full((64, 64, 3), 1, dtype=np.uint8))
    b = pipeline.ingest(np.full((64, 64, 3), 2, dtype=np.uint8))

    assert a is not None and b is not None
    assert len(pipeline.query("anything?", top_k=5)) == 2


def test_dedup_force_keeps_duplicate(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, dedup_threshold=0.95)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    pipeline.ingest(frame)
    forced = pipeline.ingest(frame, force=True)

    assert forced is not None
    assert len(pipeline.query("anything?", top_k=5)) == 2


def test_dedup_disabled_by_default(pipeline):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert pipeline.ingest(frame) is not None
    assert pipeline.ingest(frame) is not None  # no threshold -> nothing skipped


def test_query_time_window_scopes_results(pipeline):
    import time as _time

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    pipeline.ingest(frame)
    now = _time.time()

    # window covering now -> found; window entirely in the future -> empty
    assert len(pipeline.query("anything?", top_k=5, since=now - 60)) == 1
    assert len(pipeline.query("anything?", top_k=5, since=now + 3600)) == 0
    assert len(pipeline.query("anything?", top_k=5, until=now - 3600)) == 0
    assert len(pipeline.query("anything?", top_k=5, since=now - 60, until=now + 60)) == 1


def test_query_synthesize_makes_one_call_across_matches(pipeline):
    for i in range(3):
        pipeline.ingest(np.full((64, 64, 3), i, dtype=np.uint8))

    result = pipeline.query_synthesize("what happened?", top_k=3)

    assert result["answer"] == "fake synthesis of 3 images for: what happened?"
    assert len(result["matches"]) == 3
    # exactly ONE multi-image call, zero per-match calls
    assert len(pipeline.answerer.ask_many_calls) == 1
    assert len(pipeline.answerer.calls) == 0
    # every image got a timestamp label for Claude to reason about ordering
    _images, _question, labels = pipeline.answerer.ask_many_calls[0]
    assert len(labels) == 3
    assert all("captured" in label for label in labels)


def test_query_synthesize_empty_when_no_matches(pipeline):
    result = pipeline.query_synthesize("anything?", top_k=3)
    assert result == {"answer": None, "matches": []}


def test_ocr_runs_on_raw_frame_and_text_is_searchable(fake_embedder):
    ocr = _FakeOcr({1: "INVOICE total $42.00", 2: "just a wall"})
    pipeline = _make_pipeline(fake_embedder, ocr=ocr)
    pipeline.ingest(np.full((64, 64, 3), 1, dtype=np.uint8))
    pipeline.ingest(np.full((64, 64, 3), 2, dtype=np.uint8))

    # OCR read the raw frame at ingest (before any VAE encode)
    assert len(ocr.calls) == 2

    hits = pipeline.search_text("INVOICE")
    assert len(hits) == 1
    assert "INVOICE" in hits[0]["text"]
    assert hits[0]["timestamp"] is not None

    # case-insensitive: lowercase query still finds the uppercase stored text
    assert len(pipeline.search_text("invoice")) == 1
    assert len(pipeline.search_text("Invoice")) == 1

    assert pipeline.search_text("nonexistent") == []


def test_search_text_respects_time_window(fake_embedder):
    ocr = _FakeOcr({1: "receipt coffee"})
    pipeline = _make_pipeline(fake_embedder, ocr=ocr)
    pipeline.ingest(np.full((64, 64, 3), 1, dtype=np.uint8))
    now = time.time()

    assert len(pipeline.search_text("receipt", since=now - 60)) == 1
    assert len(pipeline.search_text("receipt", since=now + 3600)) == 0


def test_ocr_none_disables_text_channel(fake_embedder):
    pipeline = _make_pipeline(fake_embedder, ocr=None)
    pipeline.ingest(np.full((64, 64, 3), 1, dtype=np.uint8))
    # no OCR -> no document stored -> text search finds nothing, no crash
    assert pipeline.search_text("anything") == []


def test_recent_returns_newest_first_with_text(fake_embedder):
    ocr = _FakeOcr({1: "first note", 2: "second note", 3: "third note"})
    pipeline = _make_pipeline(fake_embedder, ocr=ocr)
    for v in (1, 2, 3):
        pipeline.ingest(np.full((64, 64, 3), v, dtype=np.uint8))

    recent = pipeline.recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["text"] == "third note"  # newest first
    assert recent[1]["text"] == "second note"
    assert all("id" in r and "timestamp" in r for r in recent)
