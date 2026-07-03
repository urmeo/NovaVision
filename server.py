"""Flask API for the NovaVision web app."""

from __future__ import annotations

import base64
import logging
import os
import random
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from novavision.pipeline import NovaVision, build_pipeline
from novavision.serving import ConcurrencyGuard, RateLimiter, env_int, resolve_host, token_ok

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("novavision.server")

MIN_TEXT, MAX_TEXT = 3, 2000

# Serve only the dedicated web asset directory, never the repo root, which would
# expose source, configs, and benchmark data.
_STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=str(_STATIC), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # cap request body

# CORS is off by default (same-origin SPA). Enable by listing origins in CORS_ORIGINS.
_origins = os.getenv("CORS_ORIGINS", "").strip()
if _origins:
    from flask_cors import CORS

    CORS(app, origins=[o.strip() for o in _origins.split(",")])

# Per-IP rate limit and a concurrency cap so a public bind cannot turn the GPU
# generate route into a trivial DoS amplifier. Both are tunable by env.
_rate_limiter = RateLimiter(env_int("NOVA_RATE_LIMIT", 30))
_gen_guard = ConcurrencyGuard(env_int("NOVA_MAX_CONCURRENCY", 2))

_pipeline: NovaVision | None = None


def pipeline() -> NovaVision:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def _emotion_list(scores: dict[str, float]):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": name, "score": round(score * 100, 1)} for name, score in ranked]


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _valid_text(data) -> tuple[str, str | None]:
    # Strip control characters before length-checking the prompt.
    text = _CONTROL_CHARS.sub(" ", (data or {}).get("text", "")).strip()
    if not (MIN_TEXT <= len(text) <= MAX_TEXT):
        return text, f"Text must be {MIN_TEXT}-{MAX_TEXT} characters."
    return text, None


def _client_ip() -> str:
    # X-Forwarded-For is client-spoofable; only trust it behind a configured proxy, else an
    # attacker could rotate the header to bypass the per-IP rate limit.
    if os.getenv("NOVA_TRUST_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limited():
    if not _rate_limiter.allow(_client_ip()):
        return jsonify({"error": "Rate limit exceeded. Try again shortly."}), 429
    return None


def _bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip()
    return request.headers.get("X-API-Token") or None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if (limited := _rate_limited()) is not None:
        return limited
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
    if not token_ok(_bearer_token()):
        return jsonify({"error": "Unauthorized."}), 401
    if (limited := _rate_limited()) is not None:
        return limited
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

    # Concurrency cap: shed load rather than queue requests behind a slow GPU job.
    if not _gen_guard.acquire():
        return jsonify({"error": "Server busy. Try again shortly."}), 429
    try:
        result = pipeline().auto_run(text, style=style, seed=seed)
    except Exception:
        logger.exception("generation failed")
        return jsonify({"error": "Generation failed."}), 500
    finally:
        _gen_guard.release()

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
    # Localhost unless NOVA_PUBLIC=1 (or a Spaces sandbox) is set; see novavision.serving.
    host = resolve_host()
    port = int(os.getenv("PORT", "8000"))
    if host == "0.0.0.0":  # noqa: S104 - explicit operator opt-in
        logger.warning("Binding 0.0.0.0 (public). Set NOVA_API_TOKEN to protect /api/generate.")
    app.run(host=host, port=port)
