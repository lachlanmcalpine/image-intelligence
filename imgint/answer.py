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

    def ask_many(
        self, images: list[np.ndarray], question: str, labels: list[str] | None = None
    ) -> str:
        """One call, one answer synthesized across all images -- for questions
        that span moments ("what did I do today") rather than identify a single
        frame. Each image is preceded by a text label (typically its capture
        timestamp) so Claude can reason about ordering and time, not just
        content. One call for N images is also cheaper and more coherent than
        N independent ask() calls whose answers never see each other.
        """
        content: list[dict] = []
        for i, image in enumerate(images):
            label = labels[i] if labels and i < len(labels) else f"Image {i + 1}"
            content.append({"type": "text", "text": f"{label}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode_image_base64(image),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"{question}\n\n"
                    "These images are the closest matches from a visual memory "
                    "search, ordered most similar first. Answer using whichever "
                    "images are relevant, say which image(s) support your answer, "
                    "and ignore matches that don't relate to the question."
                ),
            }
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise AnswerRefused(f"Claude declined to answer: {question!r}")
        return next(block.text for block in response.content if block.type == "text")


def _encode_image_base64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
