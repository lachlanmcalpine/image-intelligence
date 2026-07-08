"""Orchestrates the full visual-recall pipeline.

Write path: capture -> embed -> VAE encode -> encrypt -> store.
Read path: text query -> embed (text tower) -> transform -> search -> decrypt
-> VAE decode -> ask Claude.
"""

import time
import uuid

import numpy as np

from imgint.crypto import decrypt_latent, encrypt_latent, ensure_keys, transform_embedding


class Pipeline:
    """Takes already-constructed Embedder/VAECodec/Store/ClaudeAnswerer
    instances (dependency injection) so orchestration logic can be tested
    with fakes -- see tests/test_pipeline.py.
    """

    def __init__(self, embedder, codec, store, answerer, keys_dim: int, m=None, aes_key=None):
        self.embedder = embedder
        self.codec = codec
        self.store = store
        self.answerer = answerer
        # m/aes_key are injectable so tests (e.g. with a low-dim fake embedder)
        # never touch the real ./keys/ files -- ensure_keys() persists to a
        # fixed path keyed only by dimension, so a test-dim key would collide
        # with the real SigLIP-dim key on disk otherwise.
        if m is None or aes_key is None:
            m, aes_key = ensure_keys(keys_dim)
        self.m = m
        self.aes_key = aes_key

    def ingest(self, frame) -> str:
        """Write path: embed + encode + encrypt + store a single frame.
        Returns the record id.
        """
        vec = self.embedder.embed(frame)
        latent = self.codec.encode(frame)

        transformed_vec = transform_embedding(vec, self.m)
        nonce, ciphertext = encrypt_latent(latent.tobytes(), self.aes_key)

        record_id = str(uuid.uuid4())
        self.store.upsert(
            record_id,
            transformed_vec,
            nonce,
            ciphertext,
            extra_metadata={
                "timestamp": time.time(),
                "latent_shape": list(latent.shape),
            },
        )
        return record_id

    def query(self, question: str, top_k: int = 1) -> list[dict]:
        """Read path: embed the question (text tower), search, decrypt/decode
        the top-k matches, and ask Claude about each. Returns a list of
        {"id", "distance", "image", "answer"} dicts, best match first.
        """
        query_vec = self.embedder.embed_text(question)
        transformed_query = transform_embedding(query_vec, self.m)
        matches = self.store.query(transformed_query, top_k=top_k)

        results = []
        for match in matches:
            latent_bytes = decrypt_latent(match["nonce"], match["ciphertext"], self.aes_key)
            shape = tuple(match["metadata"]["latent_shape"])
            latent = np.frombuffer(latent_bytes, dtype=np.float32).reshape(shape)
            image = self.codec.decode(latent)
            answer = self.answerer.ask(image, question)
            results.append(
                {
                    "id": match["id"],
                    "distance": match["distance"],
                    "image": image,
                    "answer": answer,
                }
            )
        return results
