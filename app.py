"""Gradio interface for NovaVision."""

from __future__ import annotations

import logging
import threading

import gradio as gr

from novavision.pipeline import build_pipeline
from novavision.prompting import STYLE_PRESETS

logger = logging.getLogger("novavision.app")

_nova = None
_nova_lock = threading.Lock()


def _pipeline():
    # Lazy: no model or backend is built at import time.
    global _nova
    if _nova is None:
        with _nova_lock:  # double-checked: concurrent cold starts must not each load models
            if _nova is None:
                _nova = build_pipeline()
    return _nova


def generate(text: str, style: str, seed: int):
    if not text.strip():
        raise gr.Error("Please enter some text.")
    if style not in STYLE_PRESETS:
        # gradio_client bypasses the dropdown; mirror the Flask route's rejection
        # so the two entry points cannot disagree.
        raise gr.Error(f"Unknown style {style!r}.")
    try:
        result = _pipeline().auto_run(text, style=style, seed=int(seed))
    except Exception:
        logger.exception("generation failed")
        raise gr.Error("Generation failed. Please try again.") from None
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
