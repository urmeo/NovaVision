# Data card

## `tests/fixtures/affectbench_sample.csv`: test fixture (hand-authored)

56 sentences, 8 per Ekman class (anger, disgust, fear, joy, neutral, sadness, surprise).
These are **hand-authored synthetic** sentences written for this project, used **only** by
the test suite and offline demos. They are **not** GoEmotions text and must never back a
reported result. `load_benchmark` has no default path, so a run cannot fall back to them.
Columns: `text`, `emotion`.

## `content_bank.txt`: neutral content prompts

Twenty depictable, affect-neutral subjects (a city street, a tree in a field, a portrait).
The decoupled controllability track renders each subject under every intended emotion, so
recovered emotion is attributable to the conditioning rather than the scene.

**Selection criteria** (to avoid content→emotion confounds): each subject is (i) concrete and
depictable by a text-to-image model, (ii) affect-neutral on its own, no inherently happy
(wedding) or threatening (warzone) content, (iii) compatible with every emotion's mood modifier,
and (iv) varied across scene types (urban, nature, interior, person, object) so results are not
tied to one visual domain. They are hand-authored for this project.

## AffectBench: text benchmark (derived from GoEmotions)

Built on demand, not committed. Full composition, splits, and licensing are in the
datasheet ([DATASHEET.md](DATASHEET.md)):

```bash
python -m novavision.data.build_benchmark --n 100 --out data/affectbench.csv
```

## `lexicon/affect_lexicon.tsv`: affect lexicon

A small **demo** lexicon (≈80 words) with valence (-1..1) and arousal (0..1), for offline
use and tests. It is illustrative, not empirical norms. For research, replace it:

- **Warriner et al. (2013)**: `python scripts/download_lexicon.py` (freely available norms),
  then set `NOVAVISION_LEXICON` to the output.
- **NRC-VAD** (Mohammad, 2018): obtain under its terms and supply as a
  `word<TAB>valence<TAB>arousal` TSV; point `NOVAVISION_LEXICON` at it.

Scoring skips function words (with a research lexicon they would otherwise match and
distort the average) and applies a two-token negation flip to valence, so "not happy"
scores negative. Arousal is not adjusted under negation ("not calm" keeps calm's low
arousal), longer-range negation scope, double negation, and degree adverbs are not
modeled, and stopwords count toward the two-token window; coverage is the fraction of
content words matched.

## Licensing

Project code: MIT. The shipped demo lexicon is original to NovaVision (MIT). Derived/external
data inherits its source's terms (GoEmotions: Apache-2.0; Warriner norms / NRC-VAD: per their
respective licenses).
