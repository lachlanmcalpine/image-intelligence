"""Vision embedding for semantic search. The real backend lazy-imports torch/
transformers so the rest of the package stays importable without the ML stack
(mirrors face-anonymiser's deferred-import pattern for its ML backend).
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, image) -> np.ndarray:
        """Return a 1-D float32 embedding vector for a PIL Image or BGR ndarray."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Return a 1-D float32 embedding vector for a text query, in the same
        space as embed() -- used for cross-modal text-to-image search.
        """
        ...


class SigLipEmbedder:
    """Wraps google/siglip-base-patch16-224.

    Swapping to google/siglip-so400m-patch14-384 for higher retrieval accuracy
    is a one-line change to `model_id` — deferred for now, see todo.md.
    """

    MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        import torch
        from transformers import SiglipModel, SiglipProcessor

        self._torch = torch
        self.device = device
        model_id = model_id or self.MODEL_ID
        self.processor = SiglipProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id).to(device).eval()

    def embed(self, image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            import cv2
            from PIL import Image

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            # get_image_features returns a BaseModelOutputWithPooling, not a bare
            # tensor -- .pooler_output is the single pooled per-image embedding;
            # last_hidden_state is unpooled per-patch features, not what we want.
            feats = self.model.get_image_features(**inputs)
        return feats.pooler_output[0].cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(
            self.device
        )
        with self._torch.no_grad():
            # same BaseModelOutputWithPooling gotcha as get_image_features
            feats = self.model.get_text_features(**inputs)
        return feats.pooler_output[0].cpu().numpy().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
