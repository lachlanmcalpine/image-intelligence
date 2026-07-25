"""Orchestrates the full visual-recall pipeline.

Write path: capture -> embed -> [dedup gate] -> VAE encode -> encrypt -> store.
Read path: text query -> embed (text tower) -> transform -> search (optionally
time-scoped) -> decrypt -> VAE decode -> ask Claude (per-match, or one
synthesized answer across all matches).
"""

import time
import uuid
from datetime import datetime

import numpy as np

from imgint.compression import compress_latent, decompress_latent, dequantize_int8, quantize_int8
from imgint.crypto import decrypt_latent, encrypt_latent, ensure_keys, transform_embedding
from imgint.embedder import cosine_similarity
from imgint.phash import dhash, hamming


def _time_where(since: float | None, until: float | None) -> dict | None:
    """Builds a Chroma metadata filter for a [since, until] epoch-seconds
    window. Returns None when unscoped so store.query can skip the filter
    entirely (Chroma treats where={} as an error, not a no-op).
    """
    clauses = []
    if since is not None:
        clauses.append({"timestamp": {"$gte": float(since)}})
    if until is not None:
        clauses.append({"timestamp": {"$lte": float(until)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class Pipeline:
    """Takes already-constructed Embedder/VAECodec/Store/ClaudeAnswerer
    instances (dependency injection) so orchestration logic can be tested
    with fakes -- see tests/test_pipeline.py.

    Two-stage dedup, cheapest first (both compare to the last *kept* frame,
    both bypassed by force=True):
    - phash_threshold: if set, skip frames whose dHash Hamming distance to the
      last kept frame is <= it -- catches near-exact/frozen-camera duplicates
      without paying for an embedding at all (just a grayscale resize).
    - dedup_threshold: if set, skip frames whose embedding cosine similarity to
      the last kept frame is >= it -- catches semantic "same scene" duplicates
      that differ pixel-wise. Runs only on frames that survive the pHash gate.
    Either None disables that stage. Skipped frames return None instead of a
    record id, so the expensive VAE encode + upload never run and near-dupes
    don't pollute top-k retrieval. Not thread-safe by design: ingest() is only
    ever called from app.py's single worker thread.
    """

    def __init__(
        self,
        embedder,
        codec,
        store,
        answerer,
        keys_dim: int,
        m=None,
        aes_key=None,
        dedup_threshold: float | None = None,
        phash_threshold: int | None = None,
        ocr=None,
    ):
        self.embedder = embedder
        self.codec = codec
        self.store = store
        self.answerer = answerer
        self.ocr = ocr  # optional OcrReader; None disables the text channel
        self.dedup_threshold = dedup_threshold
        self.phash_threshold = phash_threshold
        self._last_kept_vec: np.ndarray | None = None
        self._last_kept_hash: int | None = None
        # m/aes_key are injectable so tests (e.g. with a low-dim fake embedder)
        # never touch the real ./keys/ files -- ensure_keys() persists to a
        # fixed path keyed only by dimension, so a test-dim key would collide
        # with the real SigLIP-dim key on disk otherwise.
        if m is None or aes_key is None:
            m, aes_key = ensure_keys(keys_dim)
        self.m = m
        self.aes_key = aes_key

    def ingest(self, frame, force: bool = False) -> str | None:
        """Write path: embed + encode + encrypt + store a single frame.
        Returns the record id, or None if the dedup gate skipped the frame.
        force=True bypasses both dedup gates (a deliberate manual capture
        should always be kept, even if the scene hasn't changed).
        """
        # Stage 1 (cheapest): perceptual-hash gate, before the ~0.3s embed.
        frame_hash = None
        if self.phash_threshold is not None:
            frame_hash = dhash(frame)
            if (
                not force
                and self._last_kept_hash is not None
                and hamming(frame_hash, self._last_kept_hash) <= self.phash_threshold
            ):
                return None

        vec = self.embedder.embed(frame)

        # Stage 2: semantic gate, on frames that survived the pHash gate.
        if (
            not force
            and self.dedup_threshold is not None
            and self._last_kept_vec is not None
            and cosine_similarity(vec, self._last_kept_vec) >= self.dedup_threshold
        ):
            return None

        # OCR runs on the RAW frame (full resolution, before the lossy VAE) --
        # small text is exactly what the autoencoder would discard, so the text
        # channel must read the original pixels, not a reconstruction.
        ocr_text = self.ocr.read_text(frame) if self.ocr is not None else None

        latent = self.codec.encode(frame)

        # int8-quantize + shuffle+zlib-compress before encrypting -- measured
        # to cost ~0.07 dB of end-to-end reconstruction quality (negligible;
        # the VAE's own lossy compression already dominates the error) while
        # cutting cloud storage to ~32% of the fp32 baseline.
        # See docs/storage-compression.md.
        quantized, scale = quantize_int8(latent)
        compressed = compress_latent(quantized)

        transformed_vec = transform_embedding(vec, self.m)
        nonce, ciphertext = encrypt_latent(compressed, self.aes_key)

        record_id = str(uuid.uuid4())
        self.store.upsert(
            record_id,
            transformed_vec,
            nonce,
            ciphertext,
            extra_metadata={
                "timestamp": time.time(),
                "latent_shape": list(latent.shape),
                "latent_scale": scale,
            },
            document=ocr_text,
        )
        self._last_kept_vec = vec
        if frame_hash is not None:
            self._last_kept_hash = frame_hash
        return record_id

    def _search_and_decode(
        self, question: str, top_k: int, since: float | None, until: float | None
    ) -> list[dict]:
        """Shared read-path front half: embed the question (text tower),
        search (optionally time-scoped), decrypt + decode each match.
        Returns [{"id", "distance", "image", "timestamp"}], best match first.
        """
        query_vec = self.embedder.embed_text(question)
        transformed_query = transform_embedding(query_vec, self.m)
        matches = self.store.query(
            transformed_query, top_k=top_k, where=_time_where(since, until)
        )

        decoded = []
        for match in matches:
            compressed = decrypt_latent(match["nonce"], match["ciphertext"], self.aes_key)
            shape = tuple(match["metadata"]["latent_shape"])
            scale = match["metadata"]["latent_scale"]
            quantized = decompress_latent(compressed, np.int8, shape)
            latent = dequantize_int8(quantized, scale)
            image = self.codec.decode(latent)
            decoded.append(
                {
                    "id": match["id"],
                    "distance": match["distance"],
                    "image": image,
                    "timestamp": match["metadata"].get("timestamp"),
                }
            )
        return decoded

    def search_text(
        self,
        text: str,
        limit: int = 10,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        """Cheap 'find what I read' path: case-insensitive substring match over
        stored OCR text, no vector search, no VLM call, and no VAE decode.
        Returns [{"id", "text", "timestamp"}], newest first. The app shows a
        per-record thumbnail (saved at ingest) rather than reconstructing the
        image. This is the core loop of the OCR-first capture prototype.
        """
        matches = self.store.get_by_text(text, limit=limit, where=_time_where(since, until))
        return [
            {"id": m["id"], "text": m["document"], "timestamp": m["metadata"].get("timestamp")}
            for m in matches
        ]

    def recent(self, limit: int = 10) -> list[dict]:
        """The most recent captures as {"id", "text", "timestamp"} -- powers
        the mobile 'recent captures' feed. Deliberately does NOT decode the
        image (the app shows a cheap per-record thumbnail saved at ingest
        instead of paying a ~1s VAE decode per item); the value here is seeing
        the read text and that processing completed, fast.
        """
        return [
            {"id": r["id"], "text": r["document"], "timestamp": r["metadata"].get("timestamp")}
            for r in self.store.get_recent(limit=limit)
        ]

    def query(
        self,
        question: str,
        top_k: int = 1,
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        """Per-match read path: each of the top-k matches gets its own
        independent Claude answer. Returns a list of {"id", "distance",
        "image", "timestamp", "answer"} dicts, best match first.
        """
        results = []
        for match in self._search_and_decode(question, top_k, since, until):
            answer = self.answerer.ask(match["image"], question)
            results.append({**match, "answer": answer})
        return results

    def query_synthesize(
        self,
        question: str,
        top_k: int = 3,
        since: float | None = None,
        until: float | None = None,
    ) -> dict:
        """Synthesized read path: ONE Claude call sees all top-k matches
        (labeled with capture timestamps) and produces one answer across
        them -- for questions that span moments ("what did I do today")
        rather than identify a single frame. Returns {"answer", "matches"};
        answer is None when nothing matched.
        """
        matches = self._search_and_decode(question, top_k, since, until)
        if not matches:
            return {"answer": None, "matches": []}

        labels = []
        for i, match in enumerate(matches):
            ts = match["timestamp"]
            if ts is not None:
                stamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                labels.append(f"Image {i + 1}, captured {stamp}")
            else:
                labels.append(f"Image {i + 1}")

        answer = self.answerer.ask_many([m["image"] for m in matches], question, labels=labels)
        return {"answer": answer, "matches": matches}
