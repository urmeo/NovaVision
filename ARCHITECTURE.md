# Architecture

NovaVision is two things sharing one core: a text-to-art **app** and an emotion-controllability
**benchmark**. Both run through the same pipeline.

## Pipeline

```
text ─► EmotionAnalyzer ─► (emotion, valence, arousal) ─► build_prompt(tier) ─► ImageBackend ─► image
```

1. **EmotionAnalyzer** (`affect/analyzer.py`) — DistilRoBERTa gives discrete-emotion scores.
   Valence/arousal come from **AffectLexicon** (`affect/lexicon.py`) over the input words,
   blended with the emotion's circumplex prior by lexical coverage. So affect is *measured
   from text*, not read from a constant per emotion.
2. **Prompt synthesis** (`prompting.py`) — three tiers: `raw` (text only), `emotion` (adds an
   emotion scene), `affect` (adds valence/arousal palette and lighting cues). The tiers are
   the ablation.
3. **Backends** (`generation/`) — a common `ImageBackend` interface with three
   implementations: `null` (deterministic, offline, for tests), `diffusers` (local, seedable),
   `hf-api` (hosted). Selected by `BACKEND`.

## Evaluation

```
image ─► CLIPAffect.recover ─► (recovered emotion, valence, arousal)
intended emotion ◄── compare ──► recovered emotion
```

`eval/clip_affect.py` runs CLIP zero-shot over a per-emotion template ensemble (recovery) and
over valence/arousal anchor prompts (affect probing). `eval/metrics.py` computes accuracy,
macro-F1, confusion, and Pearson correlation — pure numpy, fully unit-tested.
`experiments/run.py` ties it together over the benchmark and writes `results.json` + figures.
To isolate controllability from upstream classifier error, the experiment conditions on the
**gold** emotion and correlates recovered affect against the **continuous** text-grounded
valence/arousal; text-classification accuracy is reported separately.

## Data

`novavision/data/build_benchmark.py` maps single-label GoEmotions to the seven Ekman
categories (`taxonomy.GOEMOTIONS_TO_EKMAN`) and samples a balanced set. A separate
hand-authored sample and a demo lexicon ship in `data/` for offline use (see `data/README.md`).

## Design notes

- **Heavy deps are lazy.** Importing the package needs only numpy/pillow; torch, transformers,
  diffusers, and CLIP are imported inside the functions that use them, so the deterministic
  core and its tests run anywhere and in CI.
- **Seeded and reproducible.** The same seed is used per item across tiers for paired
  comparison; the null backend is deterministic for tests.
- **One source of truth.** `server.py`, `app.py`, and `experiments/run.py` are thin adapters
  over `novavision/`; there is no duplicated logic.
