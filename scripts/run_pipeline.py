"""Milestone 7 acceptance test: the real end-to-end CLI.

Usage:
    python scripts/run_pipeline.py --mode capture
    python scripts/run_pipeline.py --mode query "did I see my wallet today?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import cv2

from imgint.answer import ClaudeAnswerer
from imgint.capture import capture_frame
from imgint.codec import SdxlVaeCodec
from imgint.embedder import SigLipEmbedder
from imgint.pipeline import Pipeline
from imgint.store import Store, make_client

EMBED_DIM = 768  # google/siglip-base-patch16-224 pooled output size


def build_pipeline() -> Pipeline:
    embedder = SigLipEmbedder()
    codec = SdxlVaeCodec()
    store = Store(make_client("http"))
    answerer = ClaudeAnswerer()
    return Pipeline(embedder, codec, store, answerer, keys_dim=EMBED_DIM)


def run_capture(pipeline: Pipeline, camera_index: int) -> None:
    print(f"capturing a frame from camera index {camera_index}...")
    frame = capture_frame(index=camera_index)
    record_id = pipeline.ingest(frame)
    print(f"ingested frame -> record '{record_id}'")


def run_query(pipeline: Pipeline, question: str, top_k: int) -> None:
    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)

    print(f"querying: {question!r}")
    results = pipeline.query(question, top_k=top_k)
    if not results:
        print("no matches found")
        return

    for i, result in enumerate(results):
        print(f"\n--- match {i + 1} (id={result['id']}, distance={result['distance']:.4f}) ---")
        print(result["answer"])
        out_path = out_dir / f"query_result_{i + 1}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
        print(f"reconstructed image -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "query"], required=True)
    parser.add_argument("question", nargs="?", help="required for --mode query")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    pipeline = build_pipeline()

    if args.mode == "capture":
        run_capture(pipeline, args.camera_index)
    elif args.mode == "query":
        if not args.question:
            parser.error("query mode requires a question argument")
        run_query(pipeline, args.question, args.top_k)
