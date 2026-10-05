# Contributing

## Setup and checks

Python 3.9+ (CI: 3.9–3.12); Node 22 for browser-script tests.

```bash
python -m pip install -e ".[dev,research]" mypy pytest-cov
python -m pytest --cov=novavision
node --test tests/test_web_ui.js
ruff check novavision tests scripts server.py
ruff format --check novavision tests scripts server.py
mypy novavision
bash scripts/smoke_server.sh
```

Keep heavy imports lazy. Add regression coverage for changed behavior. Use short commit titles: `updated frontend`, `updated evaluation`, `updated outputs`.

## App

```bash
python -m pip install -e ".[app,ml]"
NOVA_BACKEND=diffusers python server.py
```

Open `http://127.0.0.1:8000`. Models download on first use. `NOVA_BACKEND=null` produces synthetic test images; text analysis still requires its classifier. Hosted generation uses `NOVA_BACKEND=hf-api` and `HF_TOKEN`.

For a public deployment, set `NOVA_API_TOKEN`, use TLS and one worker:

```bash
NOVA_BACKEND=diffusers python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind 127.0.0.1:8000 server:app
```

Limits are per process: `NOVA_RATE_LIMIT=30` requests/minute/IP, `NOVA_MAX_CONCURRENCY=2`. Trust forwarded headers only behind a configured proxy (`NOVA_TRUST_PROXY=1`). Set `CORS_ORIGINS` only for allowed origins. Request bodies are limited to 32 KiB.

## Research commands

Recompute published summaries; original records stay intact:

```bash
python scripts/resummarize.py
python scripts/report.py
python scripts/correct_recovery.py
python scripts/power_analysis.py
python scripts/compare_probes.py outputs/results/probe_validation_scene.json outputs/results/probe_validation_scene_l14.json
python scripts/make_submission.py --system "system name"
```

Generated reports go to `outputs/generated/`. Submissions use [the schema](benchmark/submission.schema.json).

New model runs write separately from published results:

```bash
python -m pip install -e ".[ml,research]"
python -m novavision.experiments.run --backend diffusers --contents 2 --seeds 1 --width 256 --height 256 --save-images --out outputs/generated/pilot
python -m novavision.data.build_benchmark --n 100 --out outputs/generated/affectbench.csv
python -m novavision.experiments.run --backend diffusers --track text --benchmark outputs/generated/affectbench.csv --seeds 3 --out outputs/generated/text
python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion --n 400 --split train --seed 0 --out outputs/generated/probe_validation.json
```

`--diffusion-model`, `--probe`, `--probe-model` and `--clip-model` select alternatives. Resume only unchanged configurations with `--resume`. The historical pilot lacks complete provenance; new renders do not reproduce its original images.

## Data and ratings

- `novavision/data/content_bank.txt`: 20 hand-authored neutral subjects. `tests/fixtures/affectbench_sample.csv`: 56 synthetic sentences, for tests only. AffectBench is built on demand from deduplicated, single-label GoEmotions test data.
- The demo lexicon is not empirical norms. Use `python scripts/download_lexicon.py` or a licensed TSV; set `NOVAVISION_LEXICON=outputs/generated/warriner.tsv`. Negation affects valence over two tokens; arousal and longer scopes are unchanged.
- Save originals with `--save-images`; then use `python -m novavision.eval.human_study build --results outputs/generated/pilot`. Verified images, blank ratings and class counts go to `outputs/generated/human_study/`. Keep `key.csv` hidden. Recruit at least three independent raters; blanks mean “cannot tell.” Report per-rater kappa, inter-rater agreement and realized class counts.
- Preserve measurements, controls and provenance. Never replace published records with fixture results. A partial correction or binomial power simulation cannot validate a seven-class recovery claim.

## Private security reports

Use [a private advisory](https://github.com/urmeo/NovaVision/security/advisories/new). Include affected version, reproduction steps and impact; omit credentials and personal data. Keep vulnerability details private until a fix is available. Treat contributors respectfully.
