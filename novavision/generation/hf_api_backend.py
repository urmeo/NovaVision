"""HuggingFace Inference API backend (optional, non-deterministic)."""

from __future__ import annotations

import os

from PIL import Image

from novavision.generation.base import ImageBackend


class HFApiBackend(ImageBackend):
    """Generates via the hosted HuggingFace Inference API."""

    name = "hf-api"

    def __init__(self, model_id: str = "stabilityai/sd-turbo", token: str | None = None):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not set")
        self.model_id = model_id
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(token=self.token)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        return self.client.text_to_image(prompt, model=self.model_id, width=width, height=height)
