# Data card

## `affectbench_sample.csv` — offline sample (hand-authored)

56 sentences, 8 per Ekman class (anger, disgust, fear, joy, neutral, sadness, surprise).
These are **hand-authored synthetic** sentences written for this project, used for offline
demos and the test suite. They are **not** GoEmotions text. Columns: `text`, `emotion`.

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
