"""Milestone 6 verification: the first true cloud integration point. Capture
a real frame, embed it, VAE-encode it, encrypt both, upsert into the real
Railway-hosted Chroma instance, query it back, decrypt, VAE-decode, and save
the reconstructed image -- proving stages 1-4 work end to end through a real
network round trip.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np

from imgint.capture import capture_frame
from imgint.codec import SdxlVaeCodec
from imgint.crypto import encrypt_latent, decrypt_latent, ensure_keys, transform_embedding
from imgint.embedder import SigLipEmbedder
from imgint.store import Store, make_client

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)

    print("capturing a frame from camera index 1...")
    frame_bgr = capture_frame(index=1)

    print("loading SigLIP base + SDXL VAE...")
    embedder = SigLipEmbedder()
    codec = SdxlVaeCodec()

    vec = embedder.embed(frame_bgr)
    latent = codec.encode(frame_bgr)
    m, aes_key = ensure_keys(dim=vec.shape[0])

    transformed_vec = transform_embedding(vec, m)
    nonce, ciphertext = encrypt_latent(latent.tobytes(), aes_key)

    print(f"connecting to Railway-hosted Chroma at {os.environ['CHROMA_HOST']}...")
    client = make_client("http")
    store = Store(client)

    record_id = "dev-store-query-test"
    store.upsert(record_id, transformed_vec, nonce, ciphertext, extra_metadata={"source": "dev_store_query"})
    print(f"upserted record '{record_id}' into the real cloud Chroma instance")

    matches = store.query(transformed_vec, top_k=1)
    assert matches, "expected at least one match back from Chroma"
    match = matches[0]
    print(f"queried back: id={match['id']} distance={match['distance']}")
    assert match["id"] == record_id, "expected the query to return the record we just upserted"

    recovered_latent_bytes = decrypt_latent(match["nonce"], match["ciphertext"], aes_key)
    recovered_latent = np.frombuffer(recovered_latent_bytes, dtype=np.float32).reshape(latent.shape)
    reconstructed = codec.decode(recovered_latent)

    out_path = out_dir / "store_query_roundtrip.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
    print(f"PASS: full cloud round trip (encrypt -> upsert -> query -> decrypt -> decode) -> {out_path}")
