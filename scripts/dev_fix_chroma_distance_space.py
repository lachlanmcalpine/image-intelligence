"""One-off migration: the "frames" Chroma collection was created without an
explicit hnsw:space, so it silently defaulted to L2 (squared Euclidean)
instead of cosine. Since the whole cross-modal search design (the
orthogonal-matrix transform in imgint/crypto.py) assumes cosine-similarity
ranking, and SigLIP embeddings aren't unit-normalized, L2 nearest-neighbor
search was ranking on the wrong thing -- confirmed live via an anomalously
large distance (643.5) and a wrong top-1 match.

Chroma fixes hnsw:space at collection-creation time and ignores metadata
changes on an already-existing collection, so the fix requires recreating
the collection. This script:
  1. Backs up every record (id, embedding, metadata) to a local JSON file.
  2. Deletes and recreates the "frames" collection with hnsw:space=cosine.
  3. Re-inserts every record unchanged -- no photos are lost, only the
     distance metric changes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from imgint.store import COLLECTION_NAME, make_client

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKUP_PATH = REPO_ROOT / "out" / "frames_collection_backup.json"

if __name__ == "__main__":
    client = make_client("http")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)

    print(f"current collection metadata: {collection.metadata}")
    count = collection.count()
    print(f"fetching all {count} records...")

    got = collection.get(include=["embeddings", "metadatas"])
    ids = got["ids"]
    embeddings = got["embeddings"]
    metadatas = got["metadatas"]
    assert len(ids) == count, f"expected {count} records, got {len(ids)}"

    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup = {
        "ids": ids,
        "embeddings": [list(map(float, e)) for e in embeddings],
        "metadatas": metadatas,
    }
    BACKUP_PATH.write_text(json.dumps(backup))
    print(f"backed up {len(ids)} records -> {BACKUP_PATH}")

    print(f"deleting collection '{COLLECTION_NAME}' (L2 space)...")
    client.delete_collection(name=COLLECTION_NAME)

    print(f"recreating '{COLLECTION_NAME}' with hnsw:space=cosine...")
    new_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"re-inserting {len(ids)} records...")
    new_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    final_count = new_collection.count()
    print(f"done. new collection count: {final_count}  metadata: {new_collection.metadata}")
    assert final_count == count, f"expected {count} records after migration, got {final_count}"
