# NovaVision: Measuring Emotion Controllability in Text-to-Image Generation via Affect Recovery

**Urme Bose** (`urme-b`)

> Draft. Table 1 and the figures are produced by the reproducible pipeline
> (`python -m novavision.experiments.run`); placeholders are shown here so no result is
> hand-written. Render the table with `python scripts/report.py`.

## Abstract

Emotional text-to-image generation is usually judged by inspection, which conflates intent with
effect. We measure emotion *controllability* directly: condition generation on an intended
emotion, recover the emotion from the generated image with CLIP, and compare it to the intent.
We release **AffectBench**, a balanced emotion-labelled benchmark derived from GoEmotions under
the Ekman taxonomy, and an **affect-conditioned prompt synthesis** that grounds generation in
valence and arousal estimated from text. Across three conditioning tiers we report recovery
accuracy, macro-F1, valence/arousal correlation, and CLIP-T. The pipeline is seeded and runs on
open models.

## 1. Introduction

Given a sentence with a known emotion, does a conditioned generator produce an image whose
recovered emotion matches the intended one, and does explicit affect grounding help over naive
prompting? We answer this with an automatic, reproducible protocol rather than human inspection.

## 2. Related work

Affect is represented with the circumplex model of valence and arousal [@russell1980]. The
benchmark reuses GoEmotions [@demszky2020] under its official Ekman grouping; text emotion is
read with a DistilRoBERTa classifier [@hartmann2022] and lexical affect with empirical norms
[@warriner2013; @mohammad2018]. Generation uses latent diffusion [@rombach2022; @sauer2023] and
evaluation uses CLIP zero-shot transfer [@radford2021].

## 3. Method

For input text $t$, valence $v$ and arousal $a$ are the lexicon-weighted mean over the affect
words in $t$, blended with the predicted emotion's circumplex prior by lexical coverage $c$:
$v = c\,v_{lex} + (1-c)\,v_{prior}$ (and likewise for $a$). Affect therefore comes from the
words while degrading gracefully when few are present.

The prompt is synthesised at three increasing tiers, which form the ablation:

- **raw** — input text and a style preset only.
- **emotion** — adds a scene template selected by the emotion.
- **affect** — also injects valence/arousal cues (palette and lighting).

## 4. AffectBench

We take single-label GoEmotions examples, map them to the seven Ekman categories, and sample a
fixed count per class (`novavision.data.build_benchmark`). A hand-authored 56-item sample
(8 per class) ships for offline runs and tests; it is synthetic, not GoEmotions text
(`data/README.md`).

## 5. Evaluation

To separate controllability from upstream classification error, each item is **conditioned on
its gold emotion**. The generated image is classified by CLIP zero-shot over a per-emotion
template ensemble, and its argmax is the recovered emotion. We report:

- **Recovery accuracy / macro-F1** — recovered vs. intended emotion.
- **Valence/arousal correlation** — Pearson $r$ between the continuous grounded affect and
  CLIP's probed affect.
- **CLIP-T** — image/text alignment, to confirm content is preserved.
- **Classification accuracy** — the upstream classifier vs. the gold label, reported separately.

Generator: SD-Turbo. Recovery: CLIP ViT-B/32. One seed per item, shared across tiers for paired
comparison; the configuration is logged to `results/results.json`.

## 6. Results

Table 1 reports each metric per tier (classification accuracy is tier-independent and reported
once); confusion matrices and valence/arousal plots are in `results/figures/`.

**Table 1.** Affect recovery by conditioning tier (generated).

| Tier | Accuracy | Macro-F1 | Valence r | Arousal r | CLIP-T |
|---|---|---|---|---|---|
| raw | – | – | – | – | – |
| emotion | – | – | – | – | – |
| affect | – | – | – | – | – |

## 7. Limitations

- CLIP recovery is a proxy for human perception; a human study is the next step.
- The emotion and affect tiers prepend a scene template, so recovery partly reflects scene
  content; the affect-minus-emotion delta isolates the continuous-grounding effect.
- The recovery prompts share vocabulary with the conditioning side; we mitigate with a template
  ensemble, but an independent classifier would strengthen validity.
- The shipped lexicon is small (use empirical norms for research), and the Ekman set omits mixed
  affect.

## 8. Reproducibility

`pip install -e ".[dev,research]" && pip install -e ".[ml]"`, then `make benchmark` and
`make reproduce` (or `make smoke`). All runs are seeded and the deterministic core is tested in
CI.

## References

See `references.bib`.
