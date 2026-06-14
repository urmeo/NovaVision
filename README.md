# NovaVision

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)

Turn a sentence into art that *feels* the way the sentence does. NovaVision detects the
emotion in text, grounds it in continuous valence/arousal, builds a conditioned prompt,
and generates an image. It also ships a small benchmark that measures whether the
generated images actually carry the intended emotion back.

Work in progress — see [ROADMAP](#roadmap).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core + research + tests
pip install -e ".[ml,app]"   # add models + web app
```

## Roadmap

- [ ] Affect lexicon + emotion analyzer
- [ ] Conditioned prompt synthesis (ablation tiers)
- [ ] Local + API image backends
- [ ] CLIP affect-recovery evaluation
- [ ] Benchmark builder
- [ ] Paper

## License

MIT
