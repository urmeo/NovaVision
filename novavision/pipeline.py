"""End-to-end text-to-image pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from novavision.affect.analyzer import EmotionAnalysis, EmotionAnalyzer
from novavision.generation import ImageBackend, get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import NEUTRAL


@dataclass
class Result:
    image: Image.Image
    prompt: str
    analysis: EmotionAnalysis
    tier: str
    seed: int
    backend: str


class NovaVision:
    """Analyze text, condition a prompt, generate an image."""

    def __init__(
        self, backend: ImageBackend | None = None, analyzer: EmotionAnalyzer | None = None
    ):
        self.backend = backend or get_backend("null")
        self.analyzer = analyzer or EmotionAnalyzer()

    def run(
        self,
        text: str,
        *,
        style: str = "artistic",
        tier: str = "affect",
        seed: int = 0,
        width: int = 512,
        height: int = 512,
    ) -> Result:
        analysis = self.analyzer.analyze(text)
        return self._render(text, analysis, style, tier, seed, width, height)

    def auto_run(
        self,
        text: str,
        *,
        style: str = "artistic",
        seed: int = 0,
        width: int = 512,
        height: int = 512,
    ) -> Result:
        """App helper: skip emotion conditioning for neutral text.

        The default ``seed=0`` is a fixed seed, not a random one: omitting it
        reproduces the same image. Pass an explicit seed (server.py randomizes
        per request) for variety.
        """
        analysis = self.analyzer.analyze(text)
        tier = "raw" if analysis.primary == NEUTRAL else "affect"
        return self._render(text, analysis, style, tier, seed, width, height)

    def _render(self, text, analysis, style, tier, seed, width, height) -> Result:
        prompt = build_prompt(
            text,
            emotion=analysis.primary,
            valence=analysis.valence,
            arousal=analysis.arousal,
            style=style,
            tier=tier,
        )
        image = self.backend.generate(
            prompt, width=width, height=height, seed=seed, negative_prompt=NEGATIVE_PROMPT
        )
        return Result(image, prompt, analysis, tier, seed, self.backend.name)


def build_pipeline(settings=None) -> NovaVision:
    """Construct the pipeline from settings, the one factory both entry points use.

    Heavy backends stay lazy (no model loads here), so importing this is cheap; the
    Flask API and the Gradio app call it instead of assembling the pipeline by hand,
    which keeps their wiring identical (one source of truth).
    """
    from novavision.config import get_settings

    cfg = settings or get_settings()
    kwargs = {"model_id": cfg.diffusion_model} if cfg.backend == "diffusers" else {}
    backend = get_backend(cfg.backend, **kwargs)
    return NovaVision(backend=backend, analyzer=EmotionAnalyzer(cfg.emotion_model))
