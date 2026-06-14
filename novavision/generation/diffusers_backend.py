"""Local diffusion backend (reproducible, seedable)."""

from __future__ import annotations

from PIL import Image

from novavision.generation.base import ImageBackend


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DiffusersBackend(ImageBackend):
    """Runs a local diffusers text-to-image pipeline."""

    name = "diffusers"

    def __init__(
        self, model_id: str = "stabilityai/sd-turbo", device: str | None = None, steps: int = 2
    ):
        self.model_id = model_id
        self.device = device or _pick_device()
        self.steps = steps
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=dtype)
            pipe = pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
        return self._pipe

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        turbo = "turbo" in self.model_id.lower()
        out = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.steps,
            guidance_scale=0.0 if turbo else 7.0,
            negative_prompt=None if turbo else negative_prompt,
            generator=generator,
        )
        return out.images[0]
