"""Text-answer generation via Claude vision. Given a reconstructed image and a
natural-language question, ask Claude to answer.

This is the one deliberate e2ee exception in the pipeline: the decrypted,
reconstructed image leaves the device to Anthropic's API at query time,
accepted as an MVP tradeoff (see todo.md). Originally planned as DeepSeek, but
DeepSeek's hosted API turned out to be text-only -- switched to Claude.
"""

import base64
import io

import numpy as np
from PIL import Image

DEFAULT_MODEL = "claude-sonnet-5"


class AnswerRefused(RuntimeError):
    """Claude's safety classifiers declined to answer (stop_reason == 'refusal')."""


class ClaudeAnswerer:
    """Accepts a pre-built `client` for dependency injection in tests (see
    tests/test_answer.py) so prompt-construction logic can be verified without
    spending real API credits.
    """

    def __init__(self, client=None, model: str = DEFAULT_MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def ask(self, image: np.ndarray, question: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _encode_image_base64(image),
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise AnswerRefused(f"Claude declined to answer: {question!r}")
        return next(block.text for block in response.content if block.type == "text")


def _encode_image_base64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
