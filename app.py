"""Gradio interface for NovaVision."""

from __future__ import annotations

import gradio as gr

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.config import get_settings
from novavision.generation import get_backend
from novavision.pipeline import NovaVision
from novavision.prompting import STYLE_PRESETS

_cfg = get_settings()
_kwargs = {"model_id": _cfg.diffusion_model} if _cfg.backend == "diffusers" else {}
_nova = NovaVision(get_backend(_cfg.backend, **_kwargs), EmotionAnalyzer(_cfg.emotion_model))


def generate(text: str, style: str, seed: int):
    if not text.strip():
        raise gr.Error("Please enter some text.")
    result = _nova.auto_run(text, style=style, seed=int(seed))
    a = result.analysis
    label = f"{a.primary} ({a.confidence:.0%}) · valence {a.valence:+.2f} · arousal {a.arousal:.2f}"
    return result.image, label, result.prompt


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="How are you feeling?", lines=2),
        gr.Dropdown(list(STYLE_PRESETS), value="artistic", label="Style"),
        gr.Number(value=0, label="Seed", precision=0),
    ],
    outputs=[gr.Image(label="Artwork"), gr.Label(label="Emotion"), gr.Textbox(label="Prompt")],
    title="NovaVision",
    description="Turn how you feel into art.",
)

if __name__ == "__main__":
    import os

    from novavision.serving import resolve_host

    # Localhost by default; binds 0.0.0.0 only on NOVA_PUBLIC=1 or a real Spaces sandbox.
    host = resolve_host()
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
