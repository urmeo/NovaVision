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
2. **Prompt synthesis** (`prompting.py`) — content stays independent of the emotion; emotion is
   layered on as a modifier over three tiers: `raw` (content only), `emotion` (adds a mood
   modifier), `affect` (adds valence/arousal palette and lighting). A `scene` floor renders the
   old fixed per-emotion template with no content, to measure pure template recognition.
3. **Backends** (`generation/`) — a common `ImageBackend` interface with three
   implementations: `null` (deterministic, offline, for tests), `diffusers` (local, seedable,
   revision-pinned), `hf-api` (hosted). Selected by `BACKEND`.

## Evaluation

```
image ─► Probe.recover ─► (recovered emotion, valence, arousal)
intended emotion ◄── compare ──► recovered emotion
```

`eval/probes.py` defines a swappable `Probe`; the default `CLIPProbe` runs CLIP zero-shot over a
per-emotion template ensemble (recovery, logit-scaled) and a graded valence/arousal anchor
ladder. `eval/metrics.py` computes accuracy, macro-F1, confusion, Pearson/Spearman, bootstrap
CIs, paired tests, and Cohen's κ — pure numpy, fully unit-tested. `experiments/run.py` runs two
tracks (`content` over the neutral bank, `text` over AffectBench) across seeds, computes the
floors and tier contrasts, and writes `results.json` (with a full provenance manifest) + figures.
`eval/human_study.py` regenerates a sampled subset for raters and scores human-vs-probe
agreement.

## Data

`novavision/data/build_benchmark.py` maps single-label GoEmotions to the seven Ekman categories
(`taxonomy.GOEMOTIONS_TO_EKMAN`), dedups, samples the **test** split, and writes a manifest with
realized per-class counts. `data/content_bank.txt` holds the neutral subjects for the decoupled
track. A hand-authored sample (test fixture) and a demo lexicon ship for offline use (see
`data/README.md`, `data/DATASHEET.md`).

## Design notes

- **Heavy deps are lazy.** Importing the package needs only numpy/pillow; torch, transformers,
  diffusers, and CLIP are imported inside the functions that use them, so the deterministic
  core and its tests run anywhere and in CI.
- **Seeded and reproducible.** The same seed is used per item across tiers for paired
  comparison; the null backend is deterministic for tests.
- **One source of truth.** `server.py`, `app.py`, and `experiments/run.py` are thin adapters
  over `novavision/`; there is no duplicated logic.
