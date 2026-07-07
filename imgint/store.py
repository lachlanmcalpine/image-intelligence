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
        self.collection = client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def upsert(
        self,
        id: str,
        transformed_vec,
        nonce: bytes,
        ciphertext: bytes,
        extra_metadata: dict | None = None,
    ) -> None:
        metadata = {
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
            "timestamp": time.time(),
            **(extra_metadata or {}),
        }
        self.collection.upsert(
            ids=[id],
            embeddings=[_as_list(transformed_vec)],
            metadatas=[metadata],
        )

    def query(self, transformed_query_vec, top_k: int = 5) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[_as_list(transformed_query_vec)], n_results=top_k
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
