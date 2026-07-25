"""Measure end-to-end latency: capture -> embed -> encode -> compress ->
encrypt -> upload to the real Railway-hosted Chroma store. Reports a
per-stage breakdown for several consecutive captures, since sustained FPS is
about steady-state performance (persistent camera, models already loaded),
not a single cold-start capture.

Model loading (SigLIP + VAE construction) and camera open+warmup are
deliberately NOT timed -- in the real app those happen once at startup.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from imgint.capture import Camera
from imgint.codec import SdxlVaeCodec
from imgint.compression import compress_latent, quantize_int8
from imgint.crypto import encrypt_latent, ensure_keys, transform_embedding
from imgint.embedder import SigLipEmbedder
from imgint.store import Store, make_client

EMBED_DIM = 768
N_FRAMES = 5

if __name__ == "__main__":
    print("loading models + opening camera (one-time startup cost, not counted below)...")
    embedder = SigLipEmbedder()
    codec = SdxlVaeCodec()
    store = Store(make_client("http"))
    m, aes_key = ensure_keys(EMBED_DIM)
    camera = Camera(index=1)

    print(f"\ntiming {N_FRAMES} consecutive capture -> searchable-in-cloud round trips...\n")

    stage_totals = {"capture": 0.0, "embed": 0.0, "vae_encode": 0.0, "compress+encrypt": 0.0, "upload": 0.0}
    record_ids = []

    for i in range(N_FRAMES):
        t0 = time.perf_counter()
        frame = camera.read()
        t1 = time.perf_counter()

        vec = embedder.embed(frame)
        t2 = time.perf_counter()

        latent = codec.encode(frame)
        t3 = time.perf_counter()

        quantized, scale = quantize_int8(latent)
        compressed = compress_latent(quantized)
        transformed_vec = transform_embedding(vec, m)
        nonce, ciphertext = encrypt_latent(compressed, aes_key)
        t4 = time.perf_counter()

        record_id = f"dev-timing-test-{i}"
        record_ids.append(record_id)
        store.upsert(
            record_id,
            transformed_vec,
            nonce,
            ciphertext,
            extra_metadata={
                "timestamp": time.time(),
                "latent_shape": list(latent.shape),
                "latent_scale": scale,
            },
        )
        t5 = time.perf_counter()

        row = {
            "capture": t1 - t0,
            "embed": t2 - t1,
            "vae_encode": t3 - t2,
            "compress+encrypt": t4 - t3,
            "upload": t5 - t4,
        }
        for k, v in row.items():
            stage_totals[k] += v
        total = t5 - t0
        print(
            f"frame {i}: capture {row['capture']*1000:6.1f}ms  embed {row['embed']*1000:6.1f}ms  "
            f"vae {row['vae_encode']*1000:6.1f}ms  compress+encrypt {row['compress+encrypt']*1000:6.1f}ms  "
            f"upload {row['upload']*1000:6.1f}ms  TOTAL {total*1000:7.1f}ms"
        )

    camera.release()
    store.collection.delete(ids=record_ids)

    print(f"\naverage over {N_FRAMES} frames:")
    grand_total = 0.0
    for k, v in stage_totals.items():
        avg = v / N_FRAMES
        grand_total += avg
        print(f"  {k:20s} {avg*1000:7.1f}ms")
    print(f"  {'TOTAL':20s} {grand_total*1000:7.1f}ms  ({1/grand_total:.2f} fps sustained)")

    print("\n(cleaned up test records)")
