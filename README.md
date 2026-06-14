# NovaVision

[![CI](https://github.com/urme-b/NovaVision/actions/workflows/ci.yml/badge.svg)](https://github.com/urme-b/NovaVision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)

Turn a sentence into art that *feels* the way the sentence does — and then check whether it
actually does. NovaVision reads the emotion in text, grounds it in continuous
valence/arousal, builds a conditioned prompt, and generates an image. It also ships a small
benchmark that measures whether generated images carry the intended emotion back.

## Demo

[![Watch the demo](https://img.youtube.com/vi/S7LZsnMPOmA/maxresdefault.jpg)](https://youtu.be/S7LZsnMPOmA)

## How it works

```
text ─► emotion classifier ─┐
                            ├─► affect grounding (valence/arousal) ─► tiered prompt ─► image
        affect lexicon  ────┘                                                            │
                                                                                         ▼
                                              CLIP affect recovery ◄── intended vs recovered emotion
```

Emotion is read with a DistilRoBERTa classifier; valence/arousal come from an affect lexicon
(blended with the emotion's circumplex prior by how many affect words the text contains), so
they are computed from the text rather than looked up from a fixed table.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + research + tests
pip install -e ".[ml,app]"     # add models + web/Gradio app
cp .env.example .env           # set BACKEND, and HF_TOKEN if using the API
```

## Run the app

```bash
python server.py               # Flask app at http://localhost:8000
python app.py                  # Gradio app at http://localhost:7860
```

Set `BACKEND=diffusers` (local, reproducible, default `stabilityai/sd-turbo`) or
`BACKEND=hf-api` with an `HF_TOKEN`. `BACKEND=null` returns placeholder images for offline
development.

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Emotion + valence/arousal for text |
| `POST` | `/api/generate` | Full pipeline → image + metadata |

```bash
curl -X POST localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"text":"I feel so grateful today"}'
```

## Research: the affect-recovery benchmark

The conditioning method has three ablation tiers — `raw`, `emotion`, `affect` — and an
automatic evaluation that asks: does a conditioned image's emotion, recovered by CLIP, match
the intended one?

```bash
make benchmark                 # build AffectBench from GoEmotions (data/affectbench.csv)
make reproduce                 # generate + evaluate across tiers -> results/results.json
python scripts/report.py       # render the metrics table for the paper
```

Metrics: affect-recovery accuracy, macro-F1, valence/arousal correlation, and CLIP-T.
Figures (confusion matrices, valence/arousal scatter) are written to `results/figures/`.
A 56-item sample benchmark and a demo lexicon ship in `data/` so everything runs offline; for
research use, swap in empirical norms with `python scripts/download_lexicon.py`.

The write-up lives in [`paper/paper.md`](paper/paper.md).

## Repo structure

```
novavision/
  taxonomy.py            # emotions, priors, GoEmotions->Ekman map
  affect/                # lexicon scoring + emotion analyzer
  prompting.py           # affect-conditioned prompt synthesis (tiers)
  generation/            # null / diffusers / hf-api backends
  pipeline.py            # text -> image
  eval/                  # CLIP recovery, metrics, figures
  data/                  # benchmark builder + loader
  experiments/run.py     # end-to-end benchmark
data/                    # sample benchmark + demo lexicon
paper/                   # write-up + references
server.py · app.py · index.html
```

## Test

```bash
make test                # deterministic core, no GPU or token needed
make lint
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT
