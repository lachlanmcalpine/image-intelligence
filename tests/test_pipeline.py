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


class _FakeAnswerer:
    def __init__(self):
        self.calls = []

    def ask(self, image, question) -> str:
        self.calls.append((image, question))
        return f"fake answer to: {question}"


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
