<div align="center">

# 🎨 NovaVision

**Turn how you feel into art — then measure whether the art actually feels that way.**

[![CI](https://github.com/urme-b/NovaVision/actions/workflows/ci.yml/badge.svg)](https://github.com/urme-b/NovaVision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

<img src="screenshots/main_interface.png" alt="NovaVision interface" width="820">

</div>

NovaVision reads the emotion in a sentence, grounds it in continuous **valence/arousal**,
builds an affect-conditioned prompt, and generates an image. It is both a **web app** and a
small **research benchmark** that asks a falsifiable question: *does a conditioned image
actually carry the intended emotion back to a viewer?*

<div align="center">
  <img src="screenshots/demo_preview.gif" alt="Demo" width="820">
  <br><em><a href="https://youtu.be/S7LZsnMPOmA">▶ Watch the full demo</a></em>
</div>

## ✨ Features

- **Emotion → art** across 7 Ekman emotions, with five visual styles.
- **Grounded affect** — valence/arousal are *computed from the text* (lexicon blended with
  circumplex priors), not looked up from constants.
- **Pluggable backends** — local `sd-turbo` (seedable, reproducible), the HF Inference API,
  or a deterministic `null` backend for offline dev/CI.
- **AffectBench** — a reproducible benchmark + automatic CLIP *affect-recovery* metric.
- **Production-checked** — 38 tests, ruff-clean, CI, Flask + gunicorn + Gradio.

## 🧠 How it works

<div align="center"><img src="screenshots/how_it_works.png" alt="Pipeline" width="760"></div>

```
text ─► emotion classifier ─┐
                            ├─► affect grounding (valence/arousal) ─► tiered prompt ─► image
        affect lexicon  ────┘                                                           │
                                                                                        ▼
                                          CLIP affect recovery ◄── intended vs recovered emotion
```

<div align="center"><img src="screenshots/emotion_analysis.png" alt="Emotion analysis" width="760"></div>

## 🚀 Quick start

```bash
git clone https://github.com/urme-b/NovaVision.git
cd NovaVision
python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev,research]"   # core + research + tests
pip install -e ".[ml,app]"         # models + web app
cp .env.example .env               # choose BACKEND (default: diffusers / sd-turbo)

python server.py                   # → http://localhost:8000
```

## 🔌 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze`  | Emotion + valence/arousal for text |
| `POST` | `/api/generate` | Full pipeline → image + metadata |

```bash
curl -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"text":"I feel grateful and joyful today"}'
# → {"primary_emotion":"joy","confidence":99.2,"valence":0.81,"arousal":0.70, ...}
```

## 🔬 Research: AffectBench

Three conditioning tiers (`raw` → `emotion` → `affect`) are evaluated by an automatic
**affect-recovery** protocol: generate an image, classify its emotion back with CLIP, and
compare to the intended label. Metrics: recovery accuracy, macro-F1, valence/arousal
correlation, CLIP-T, and (separately) text-classification accuracy.

```bash
make benchmark     # build AffectBench from GoEmotions (Ekman mapping)
make reproduce     # generate + evaluate across tiers → results/
python scripts/report.py   # render the metrics table
```

The write-up is in [`paper/paper.md`](paper/paper.md); data provenance in
[`data/README.md`](data/README.md). Everything is seeded; the deterministic core runs
offline in CI.

## 📁 Project structure

```
novavision/
  taxonomy.py          emotions, priors, GoEmotions→Ekman map
  affect/              lexicon scoring + emotion analyzer
  prompting.py         affect-conditioned prompt synthesis (tiers)
  generation/          null · diffusers · hf-api backends
  pipeline.py          text → image
  eval/                CLIP recovery · metrics · figures
  data/ · experiments/ benchmark builder · runner
data/  paper/  tests/  server.py  app.py  index.html
```

## 🛠 Tech stack

Python · HuggingFace Transformers · Diffusers (SD-Turbo) · CLIP · Flask · Gradio ·
Pydantic · pytest · ruff · Docker

## ✅ Testing

```bash
make test    # 38 tests, offline, no GPU or token
make lint
```

## 📚 Citation

See [`CITATION.cff`](CITATION.cff).

## 📄 License

[MIT](LICENSE) © urme-b
