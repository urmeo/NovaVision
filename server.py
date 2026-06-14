"""Flask API for the NovaVision web app."""

from __future__ import annotations

import base64
import logging
import os
import random
from datetime import datetime, timezone
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.config import get_settings
from novavision.generation import get_backend
from novavision.pipeline import NovaVision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("novavision.server")

MIN_TEXT, MAX_TEXT = 3, 2000

app = Flask(__name__, static_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # cap request body

# CORS is off by default (same-origin SPA). Enable by listing origins in CORS_ORIGINS.
_origins = os.getenv("CORS_ORIGINS", "").strip()
if _origins:
    from flask_cors import CORS

    CORS(app, origins=[o.strip() for o in _origins.split(",")])

_pipeline: NovaVision | None = None


def pipeline() -> NovaVision:
    global _pipeline
    if _pipeline is None:
        cfg = get_settings()
        kwargs = {"model_id": cfg.diffusion_model} if cfg.backend == "diffusers" else {}
        backend = get_backend(cfg.backend, **kwargs)
        _pipeline = NovaVision(backend=backend, analyzer=EmotionAnalyzer(cfg.emotion_model))
    return _pipeline


def _emotion_list(scores: dict[str, float]):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": name, "score": round(score * 100, 1)} for name, score in ranked]


def _valid_text(data) -> tuple[str, str | None]:
    text = (data or {}).get("text", "").strip()
    if not (MIN_TEXT <= len(text) <= MAX_TEXT):
        return text, f"Text must be {MIN_TEXT}-{MAX_TEXT} characters."
    return text, None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    text, error = _valid_text(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    try:
        a = pipeline().analyzer.analyze(text)
    except Exception:
        logger.exception("analyze failed")
        return jsonify({"error": "Analysis failed."}), 500

    return jsonify(
        {
            "success": True,
            "primary_emotion": a.primary,
            "confidence": round(a.confidence * 100, 1),
            "valence": a.valence,
            "arousal": a.arousal,
            "emotions": _emotion_list(a.scores),
        }
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True)
    text, error = _valid_text(data)
    if error:
        return jsonify({"error": error}), 400

    style = str((data or {}).get("style", "artistic")).lower()
    raw_seed = (data or {}).get("seed")
    try:
        seed = random.randint(0, 2**31 - 1) if raw_seed is None else int(raw_seed)
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer."}), 400

    try:
        result = pipeline().auto_run(text, style=style, seed=seed)
    except Exception:
        logger.exception("generation failed")
        return jsonify({"error": "Generation failed."}), 500

    buffer = BytesIO()
    result.image.save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    a = result.analysis
    return jsonify(
        {
            "success": True,
            "image": f"data:image/png;base64,{image}",
            "primary_emotion": a.primary,
            "confidence": round(a.confidence * 100, 1),
            "valence": a.valence,
            "arousal": a.arousal,
            "emotions": _emotion_list(a.scores),
            "prompt": result.prompt,
            "original_text": text,
            "style": style,
            "seed": result.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port)
