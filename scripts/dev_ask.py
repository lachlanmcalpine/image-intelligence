"""Milestone 8 verification: ask Claude a real question about a real
reconstructed image and print the answer end to end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from imgint.answer import ClaudeAnswerer
from imgint.capture import capture_frame
from imgint.codec import SdxlVaeCodec

if __name__ == "__main__":
    print("capturing a frame from camera index 1...")
    frame_bgr = capture_frame(index=1)

    print("loading SDXL VAE (madebyollin/sdxl-vae-fp16-fix)...")
    codec = SdxlVaeCodec()
    latent = codec.encode(frame_bgr)
    reconstructed = codec.decode(latent)  # RGB uint8, 256x256

    question = sys.argv[1] if len(sys.argv) > 1 else "Describe what you see in this image in one sentence."

    print(f"asking Claude Sonnet 5: {question!r}")
    answerer = ClaudeAnswerer()
    answer = answerer.ask(reconstructed, question)
    print(f"\nAnswer: {answer}")
