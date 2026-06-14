"""Flask API for the NovaVision web app."""

from __future__ import annotations

import base64
import random
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.config import get_settings
from novavision.generation import get_backend
from novavision.pipeline import NovaVision

app = Flask(__name__, static_folder=".")
CORS(app)

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


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    a = pipeline().analyzer.analyze(text)
    return jsonify({
        "success": True,
        "primary_emotion": a.primary,
        "confidence": round(a.confidence * 100, 1),
        "valence": a.valence,
        "arousal": a.arousal,
        "emotions": _emotion_list(a.scores),
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    style = data.get("style", "artistic")
    seed = data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    try:
        result = pipeline().auto_run(text, style=style.lower(), seed=int(seed))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    buffer = BytesIO()
    result.image.save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    a = result.analysis
    return jsonify({
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
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
