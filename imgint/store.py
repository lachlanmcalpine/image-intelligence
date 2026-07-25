"""Chroma vector store wrapper. Embeddings are stored in their orthogonal-
matrix-transformed form (Chroma needs to compute HNSW distances on them, so
they can't be AES-encrypted); the VAE latent is AES-GCM encrypted and stored
as opaque metadata (base64-encoded nonce + ciphertext).

Deliberately never imports imgint.crypto -- key material must never be
reachable from the code path that talks to the cloud (see tests/test_store.py).
"""

import base64
import time
from typing import Any

COLLECTION_NAME = "frames"


def make_client(mode: str = "ephemeral"):
    import chromadb

    if mode == "ephemeral":
        return chromadb.EphemeralClient()
    if mode == "http":
        import os

        host = os.environ["CHROMA_HOST"]
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        ssl = os.environ.get("CHROMA_SSL", "true").lower() == "true"
        headers = {}
        token = os.environ.get("CHROMA_AUTH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
    raise ValueError(f"unknown store mode: {mode}")


class Store:
    def __init__(self, client, collection_name: str = COLLECTION_NAME):
        self.client = client
        # hnsw:space must be explicit: Chroma defaults to L2, but
        # transform_embedding() (imgint/crypto.py) only preserves *cosine*
        # similarity under the orthogonal transform, not L2 distance --
        # SigLIP embeddings aren't unit-normalized, so L2 nearest-neighbor
        # ranks differently than cosine and silently returns the wrong
        # match. Chroma fixes hnsw:space at creation time and ignores this
        # metadata on an already-existing collection, so an existing
        # non-cosine collection needs to be recreated, not just re-opened
        # (see scripts/dev_fix_chroma_distance_space.py).
        self.collection = client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        id: str,
        transformed_vec,
        nonce: bytes,
        ciphertext: bytes,
        extra_metadata: dict | None = None,
        document: str | None = None,
    ) -> None:
        """`document` is the frame's OCR text -- stored as Chroma's document
        field (not auto-embedded, since embedding_function=None) so it's
        searchable by substring via get_by_text without a vector or VLM call.
        It is plaintext on the server: acceptable because text is far lower
        stakes than the image (and the image stays AES-encrypted), but noted
        as a deliberate exception to the "server sees nothing" posture -- see
        todo.md.
        """
        metadata = {
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
            "timestamp": time.time(),
            **(extra_metadata or {}),
        }
        kwargs: dict[str, Any] = {
            "ids": [id],
            "embeddings": [_as_list(transformed_vec)],
            "metadatas": [metadata],
        }
        # Chroma rejects empty-string documents; only attach when there's text
        if document:
            kwargs["documents"] = [document]
        self.collection.upsert(**kwargs)

    def get_by_text(self, text: str, limit: int = 10, where: dict | None = None) -> list[dict[str, Any]]:
        """Case-insensitive substring search over stored OCR text -- 'find
        frames whose visible text contains X'. No query vector, no VLM.
        Optionally scoped by a metadata `where` (e.g. a timestamp window).
        Returns newest-first.

        Filters client-side (lowercased) rather than via Chroma's
        where_document, whose $contains is case-SENSITIVE -- a real usability
        trap ("Baby" not matching stored "baby"). Fine at prototype scale;
        for a large collection this should move to a proper case-insensitive
        full-text index (pgvector tsvector / a lowercased indexed field). See
        todo.md.
        """
        needle = text.lower()
        rows = self._get_all(where)
        matches = [r for r in rows if r["document"] and needle in r["document"].lower()]
        matches.sort(key=lambda m: m["metadata"].get("timestamp", 0), reverse=True)
        return matches[:limit]

    def get_recent(self, limit: int = 10, where: dict | None = None) -> list[dict[str, Any]]:
        """The most recent records (newest first), regardless of text -- powers
        the 'recent captures' feed so you can see what was just captured and
        what text was read from it."""
        rows = self._get_all(where)
        rows.sort(key=lambda m: m["metadata"].get("timestamp", 0), reverse=True)
        return rows[:limit]

    def _get_all(self, where: dict | None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"include": ["metadatas", "documents"]}
        if where is not None:
            kwargs["where"] = where
        result = self.collection.get(**kwargs)
        documents = result.get("documents") or [None] * len(result["ids"])
        rows = []
        for id_, metadata, document in zip(result["ids"], result["metadatas"], documents):
            rows.append(
                {
                    "id": id_,
                    "nonce": base64.b64decode(metadata["nonce_b64"]),
                    "ciphertext": base64.b64decode(metadata["ciphertext_b64"]),
                    "document": document,
                    "metadata": metadata,
                }
            )
        return rows

    def query(
        self, transformed_query_vec, top_k: int = 5, where: dict | None = None
    ) -> list[dict[str, Any]]:
        """`where` is a Chroma metadata filter (e.g. a timestamp range built by
        pipeline._time_where) applied *before* nearest-neighbor ranking -- time
        is the primary axis for most questions ("what did I do today"), so
        filtering must scope the search, not post-filter its results.
        """
        kwargs: dict[str, Any] = {}
        if where is not None:
            kwargs["where"] = where
        result = self.collection.query(
            query_embeddings=[_as_list(transformed_query_vec)], n_results=top_k, **kwargs
        )
        ids = result["ids"][0]
        metadatas = result["metadatas"][0]
        distances = result.get("distances", [[None] * len(ids)])[0]

        matches = []
        for id_, metadata, distance in zip(ids, metadatas, distances):
            matches.append(
                {
                    "id": id_,
                    "nonce": base64.b64decode(metadata["nonce_b64"]),
                    "ciphertext": base64.b64decode(metadata["ciphertext_b64"]),
                    "distance": distance,
                    "metadata": metadata,
                }
            )
        return matches


def _as_list(vec) -> list[float]:
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)
