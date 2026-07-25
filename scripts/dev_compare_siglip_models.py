"""Phase 3a: SigLIP base vs SO400M, measured side by side on real captured
photos -- real embed latency and real retrieval/discrimination quality, not
a parameter-count guess. This hardware has repeatedly behaved non-linearly
this session (e.g. ONNX being *slower* than PyTorch), so the ~4-5x
parameter-ratio latency guess isn't trusted here without measuring it.

Report only -- no pipeline change. If SO400M is adopted, it's a one-line
`SigLipEmbedder.MODEL_ID` change (see todo.md).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from imgint.embedder import SigLipEmbedder, cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTOS = [
    REPO_ROOT / "static" / "captures" / "e2f8a41ab1ad4157875150f597be1283.jpg",  # empty room, window, hanger
    REPO_ROOT / "static" / "captures" / "c9b9a747e5324675aac8c72716f53293.jpg",  # person in black hoodie, phone
]
PHOTO_LABELS = ["empty_room", "person_hoodie"]

QUERIES = [
    "a person wearing a black hoodie holding a phone",
    "an empty room with a window and clothes on a hanger",
    "a dog running on a beach",
    "a plate of spaghetti",
]
# which query index is the "correct" match for each photo, by position in PHOTOS
EXPECTED_MATCH = [1, 0]  # empty_room -> query[1], person_hoodie -> query[0]

N_TIMED_RUNS = 5


def time_embed(embedder, frame, n=N_TIMED_RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        embedder.embed(frame)
        times.append(time.perf_counter() - t0)
    # steady-state: drop the first call (cold caches)
    steady = times[1:]
    return times, sum(steady) / len(steady)


def run_model(label, model_id):
    print(f"\n=== {label} ({model_id}) ===")
    print("loading model...")
    t_load0 = time.perf_counter()
    embedder = SigLipEmbedder(model_id=model_id)
    load_s = time.perf_counter() - t_load0
    print(f"  load time: {load_s:.1f}s (first run also downloads weights)")

    frame = cv2.imread(str(PHOTOS[0]))
    times, avg_steady = time_embed(embedder, frame)
    times_ms = [f"{t*1000:.0f}" for t in times]
    print(f"  embed latency (5 runs, 1st is cold): {times_ms} ms")
    print(f"  steady-state avg (excl. 1st): {avg_steady*1000:.1f}ms  ({1/avg_steady:.2f} fps)")

    image_vecs = []
    for photo_path in PHOTOS:
        frame = cv2.imread(str(photo_path))
        image_vecs.append(embedder.embed(frame))

    query_vecs = [embedder.embed_text(q) for q in QUERIES]
    dim = image_vecs[0].shape[0]
    print(f"  embedding dim: {dim}  (fp32 storage: {dim*4} bytes)")

    print(f"  {'photo':16s} " + " ".join(f"q{i}" for i in range(len(QUERIES))) + "   correct-margin")
    margins = []
    for i, label_i in enumerate(PHOTO_LABELS):
        sims = [cosine_similarity(image_vecs[i], qv) for qv in query_vecs]
        sims_str = " ".join(f"{s:+.3f}" for s in sims)
        expected_idx = EXPECTED_MATCH[i]
        best_other = max(s for j, s in enumerate(sims) if j != expected_idx)
        margin = sims[expected_idx] - best_other
        margins.append(margin)
        print(f"  {label_i:16s} {sims_str}   {margin:+.3f}")

    return {
        "label": label,
        "load_s": load_s,
        "steady_state_ms": avg_steady * 1000,
        "dim": dim,
        "margins": margins,
    }


if __name__ == "__main__":
    for photo in PHOTOS:
        if not photo.exists():
            print(f"missing expected real photo: {photo}")
            sys.exit(1)

    results = []
    results.append(run_model("SigLIP base", "google/siglip-base-patch16-224"))
    results.append(run_model("SigLIP SO400M", "google/siglip-so400m-patch14-384"))

    print("\n\n=== SUMMARY ===")
    print(f"{'model':16s} {'load(s)':>8s} {'embed(ms)':>10s} {'dim':>6s} {'bytes':>8s} {'min margin':>12s}")
    for r in results:
        print(
            f"{r['label']:16s} {r['load_s']:8.1f} {r['steady_state_ms']:10.1f} "
            f"{r['dim']:6d} {r['dim']*4:8d} {min(r['margins']):+12.3f}"
        )

    base, so400m = results
    slowdown = so400m["steady_state_ms"] / base["steady_state_ms"]
    storage_delta = (so400m["dim"] - base["dim"]) * 4
    print(f"\nSO400M is {slowdown:.2f}x the embed latency of base on this hardware.")
    print(f"Embedding storage delta: +{storage_delta} bytes/image (dim {base['dim']} -> {so400m['dim']}).")
    print("(latent/int8 storage, which dominates the ~8KB/image total, is unaffected by this choice.)")
