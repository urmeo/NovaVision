# Data card

## `tests/fixtures/affectbench_sample.csv` — test fixture (hand-authored)

56 sentences, 8 per Ekman class (anger, disgust, fear, joy, neutral, sadness, surprise).
These are **hand-authored synthetic** sentences written for this project, used **only** by
the test suite and offline demos. They are **not** GoEmotions text and must never back a
reported result — `load_benchmark` has no default path, so a run cannot fall back to them.
Columns: `text`, `emotion`.

## `content_bank.txt` — neutral content prompts

Twenty depictable, affect-neutral subjects (a city street, a tree in a field, a portrait).
The decoupled controllability track renders each subject under every intended emotion, so
recovered emotion is attributable to the conditioning rather than the scene.

## AffectBench — full benchmark (derived from GoEmotions)

The full benchmark is built on demand, not committed:

```bash
python -m novavision.data.build_benchmark --n 100 --out data/affectbench.csv
```

It takes single-label examples from **GoEmotions** (Demszky et al., 2020), maps the 27
labels to the seven Ekman categories using the dataset's **official Ekman grouping**
(`novavision/taxonomy.py:GOEMOTIONS_TO_EKMAN`), and samples a balanced set per class.
GoEmotions is released under **Apache-2.0**; cite Demszky et al. (2020) when using AffectBench.

## `lexicon/affect_lexicon.tsv` — affect lexicon

A small **demo** lexicon (≈80 words) with valence (−1..1) and arousal (0..1), for offline
use and tests. It is illustrative, not empirical norms. For research, replace it:

- **Warriner et al. (2013)** — `python scripts/download_lexicon.py` (freely available norms),
  then set `NOVAVISION_LEXICON` to the output.
- **NRC-VAD** (Mohammad, 2018) — obtain under its terms and supply as a
  `word<TAB>valence<TAB>arousal` TSV; point `NOVAVISION_LEXICON` at it.

## Licensing

Project code: MIT. Derived data inherits its source's terms (GoEmotions: Apache-2.0;
Warriner norms / NRC-VAD: per their respective licenses).
