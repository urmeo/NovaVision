# NovaVision

NovaVision turns a sentence into an image that reflects its emotional tone, and provides a
benchmark for measuring whether the generated image conveys the intended emotion. It runs both
as a web app and as a command-line research pipeline.

## How it works

1. A text classifier detects the dominant emotion across seven Ekman categories.
2. Valence and arousal are estimated from the text with an affect lexicon, blended with the
   detected emotion's circumplex prior in proportion to how many affect words appear.
3. A prompt is composed at one of three conditioning levels — raw text, emotion scene, or
   affect-grounded — and passed to an image backend.
4. For evaluation, CLIP classifies the emotion of the generated image, which is compared
   against the intended label (affect recovery).

## Requirements

- Python 3.9 or newer.
- For local image generation, enough memory to run Stable Diffusion Turbo. A GPU or Apple
  Silicon (MPS) is recommended; CPU works but is slow.

## Installation

```bash
git clone https://github.com/urme-b/NovaVision.git
cd NovaVision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]"   # library, tests, and benchmark tools
pip install -e ".[ml,app]"         # add models and the web app
```

## Usage

Create a local environment file and choose a backend:

```bash
cp .env.example .env
```

Run the web app:

```bash
python server.py   # Flask app on http://localhost:8000
python app.py      # Gradio app on http://localhost:7860
```

### API

- `POST /api/analyze` returns the detected emotion with valence and arousal.
- `POST /api/generate` runs the full pipeline and returns a generated image with metadata.

```bash
curl -X POST localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "I feel grateful and joyful today"}'
```

## Configuration

Configuration is read from environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND` | Image backend: `null`, `diffusers`, or `hf-api` | `null` |
| `DIFFUSION_MODEL` | Model id for the `diffusers` backend | `stabilityai/sd-turbo` |
| `HF_TOKEN` | API token, required only for `hf-api` | unset |
| `NOVAVISION_LEXICON` | Path to a full affect lexicon TSV | bundled demo lexicon |
| `HOST`, `PORT` | Bind address for the Flask server | `127.0.0.1`, `8000` |
| `CORS_ORIGINS` | Comma-separated allowed origins | same-origin only |

## Benchmark

The research pipeline builds a balanced benchmark, generates images across the three
conditioning tiers, and scores affect recovery.

```bash
make benchmark            # build AffectBench from GoEmotions
make reproduce            # generate and evaluate, writing to results/
python scripts/report.py  # render the metrics table
```

A small, hand-authored sample benchmark and a demo lexicon ship in `data/` so the pipeline
runs without downloads; see `data/README.md` for provenance. The method and evaluation are
described in `paper/paper.md`.

## Development

```bash
make test   # test suite (offline, no GPU or token required)
make lint   # ruff checks and format verification
```

## Project layout

```
novavision/   core library: taxonomy, affect, prompting, generation, eval, pipeline
data/         sample benchmark and demo lexicon
paper/        method write-up and references
tests/        test suite
server.py     Flask API and static frontend
app.py        Gradio interface
index.html    single-page web client
```

## Contributing

Issues and pull requests are welcome. Run `make lint` and `make test` before submitting.
Report security issues as described in `SECURITY.md`.

## License

Released under the MIT License. See `LICENSE`.
