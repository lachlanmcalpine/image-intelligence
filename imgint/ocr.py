"""Text extraction (OCR) as a separate channel from the image codec.

Text is the highest-value thing a wearable/phone camera captures for a
knowledge worker (whiteboards, docs, receipts, cards, screens, labels), and
it must NOT go through the lossy VAE -- small text is exactly what an
autoencoder discards. So OCR runs on the raw full-resolution frame at capture
time and its output is stored as plain searchable text, independent of however
aggressively the image itself is compressed. A text query then hits a string
index (near-free, exact) instead of a vector search or a VLM call.

Two backends:
- ClaudeOcrReader (default): a vision LLM read. Handles handwriting, angled/
  faint receipts, stylized labels -- the real-world captures that traditional
  OCR mangles (proven on real phone captures: RapidOCR read a handwritten
  whiteboard as "baby J Job -fum on dil"; Claude read it exactly). Costs a
  fraction of a cent + a few seconds per kept frame -- fine for tap-to-capture,
  too expensive for continuous 5fps (see todo.md). Also returns a short
  description, so search works for "receipt"/"whiteboard", not just literal text.
- RapidOcrReader: ONNX-based, local, free, fast -- great on clean printed text,
  poor on handwriting/messy scenes. The cheap fallback.

Lazy-imported so the rest of the package stays importable without the heavy
deps, mirroring embedder.py/codec.py.
"""

import base64
import io
from typing import Protocol

import numpy as np

# Anthropic's per-image limit is 10 MB and it recommends <=1568px long edge;
# downscale + JPEG (never PNG -- a full photo as PNG blows past 10 MB).
_MAX_EDGE = 1568

_OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written, including "
    "handwriting. Then on a new line write 'Description:' followed by one short "
    "sentence describing what this is (e.g. a receipt, a whiteboard note, a "
    "business card, a screen). If there is no text at all, just give the Description."
)


class OcrReader(Protocol):
    def read_text(self, frame) -> str:
        """Return all text detected in a BGR frame as a single string
        (empty string if none), for storage as a searchable document.
        """
        ...


def _encode_jpeg_b64(frame_bgr: np.ndarray, max_edge: int = _MAX_EDGE) -> str:
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    w, h = image.size
    scale = max_edge / max(w, h)
    if scale < 1:
        image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ClaudeOcrReader:
    """Reads text (and a one-line description) from a frame via Claude vision.
    Accepts a pre-built `client` for dependency injection in tests.
    """

    def __init__(self, client=None, model: str = "claude-sonnet-5"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def read_text(self, frame: np.ndarray) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _encode_jpeg_b64(frame),
                            },
                        },
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ],
        )
        # a refusal or a non-text-only response shouldn't crash ingest -- just
        # store no text for that frame (it stays image/embedding-searchable)
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return next((b.text for b in response.content if b.type == "text"), "")


class RapidOcrReader:
    """Wraps RapidOCR. Loads the detection+recognition models once on
    construction (eager, like the other real backends) so capture-time OCR
    doesn't pay the load cost per frame.
    """

    def __init__(self, min_confidence: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()
        self.min_confidence = min_confidence

    def read_text(self, frame: np.ndarray) -> str:
        result, _elapsed = self._ocr(frame)
        if not result:
            return ""
        # each result row is [box, text, confidence]; RapidOCR returns the
        # confidence as a STRING (e.g. '0.7367'), so cast before comparing.
        # Keep confident text, joined in reading order with spaces.
        parts = [text for _box, text, conf in result if float(conf) >= self.min_confidence]
        return " ".join(parts)
