# NovaVision: Measuring Emotion Controllability in Text-to-Image Generation via Affect Recovery

**Urme Bose** (`urme-b`)

> Draft. The numbers in Table 1 and the figures are produced by the reproducible
> pipeline (`python -m novavision.experiments.run`); this file ships with placeholders
> so no result is hand-written. Render the table with `python scripts/report.py`.

## Abstract

Text-to-image models are increasingly steered by emotional language, yet it is rarely
checked whether the resulting images actually *carry* the intended emotion back to an
observer. We study emotion controllability directly. We contribute (i) **AffectBench**, a
balanced benchmark of emotion-labelled prompts derived from GoEmotions under the Ekman
taxonomy; (ii) an **affect-conditioned prompt synthesis** method that grounds generation in
valence/arousal estimated from text rather than fixed per-class constants; and (iii) an
**affect-recovery protocol** that classifies the emotion of each generated image with CLIP
and compares it to the intended label. Across three conditioning tiers — raw text, discrete
emotion, and affect-grounded — we report recovery accuracy, macro-F1, valence/arousal
correlation, and CLIP-T. The whole pipeline is seeded and runs on open models.

## 1. Introduction

Affective text-to-image generation is usually evaluated by inspection: a prompt is written,
an image appears, and it "looks happy". This conflates *intent* with *effect*. We ask a
falsifiable question instead: given a sentence with a known emotion, does a conditioned
generator produce an image whose recovered emotion matches the intended one, and does
explicit affect grounding help over naive prompting?

## 2. Related Work

Our affect representation follows the circumplex model of valence and arousal [@russell1980].
The benchmark reuses GoEmotions [@demszky2020] under its official Ekman grouping. Text emotion
is read with a DistilRoBERTa classifier [@hartmann2022]; lexical affect uses empirical norms
[@warriner2013; @mohammad2018]. Generation uses latent diffusion [@rombach2022], optionally
its distilled real-time variant [@sauer2023], and evaluation uses CLIP zero-shot transfer
[@radford2021].

## 3. Method

**Affect grounding.** For input text $t$ we estimate valence $v$ and arousal $a$ as a
lexicon-weighted mean over the affect words in $t$, then blend with the predicted emotion's
circumplex prior by lexical coverage $c$: $v = c\,v_{lex} + (1-c)\,v_{prior}$ (likewise for
$a$). This grounds affect in the actual words while staying robust when few are present.

**Conditioning tiers.** We synthesise the generation prompt at three increasing levels:

- **raw** — the input text plus a style preset (no emotion signal).
- **emotion** — adds a scene template selected by the emotion.
- **affect** — additionally injects valence/arousal cues (palette and lighting).

The tiers form an ablation isolating the contribution of discrete-emotion conditioning and
of continuous affect grounding.

## 4. AffectBench

The full benchmark is built from single-label GoEmotions examples, mapped to the seven Ekman
categories (anger, disgust, fear, joy, neutral, sadness, surprise) and sampled to a fixed
count per class (`novavision.data.build_benchmark`). A separate, **hand-authored** 56-item
sample (8 per class) ships in the repository for offline runs and tests; it is synthetic, not
GoEmotions text. See `data/README.md`.

## 5. Evaluation: affect recovery

To isolate controllability from upstream text-classification error, the experiment
**conditions on the gold emotion** (the benchmark label) and grounds valence/arousal in the
text. Each generated image is passed to a CLIP zero-shot classifier — an ensemble of
templates per emotion, averaged — and its argmax is the *recovered* emotion. We report:

- **Recovery accuracy / macro-F1** — recovered vs. intended (conditioned) emotion.
- **Valence/arousal correlation** — Pearson $r$ between the *continuous grounded* affect we
  conditioned on and CLIP's probed affect (not a per-class constant).
- **CLIP-T** — image/text alignment, to confirm content is not lost when conditioning.
- **Text-classification accuracy** — the upstream classifier vs. the gold label, reported
  separately so it is not folded into the controllability number.

## 6. Experimental setup

Generator: SD-Turbo (open, seedable). Recovery: CLIP ViT-B/32. The same seed is used per item
across tiers for paired comparison. Full configuration is recorded in `results/results.json`.

## 7. Results

Table 1 reports each metric per conditioning tier; text-classification accuracy is reported
once (it is tier-independent). Per-tier confusion matrices and valence/arousal scatter plots
are in `results/figures/`.

**Table 1.** Affect recovery by conditioning tier (generated).

| Tier | Accuracy | Macro-F1 | Valence r | Arousal r | CLIP-T |
|---|---|---|---|---|---|
| raw | – | – | – | – | – |
| emotion | – | – | – | – | – |
| affect | – | – | – | – | – |

## 8. Discussion

The paired design lets us read off whether discrete-emotion conditioning improves recovery
over raw prompting, and whether affect grounding adds further gains without hurting CLIP-T.
Confusion matrices show which emotions are reliably conveyed and which collapse (commonly
fear/sadness and surprise/joy).

## 9. Limitations

- **CLIP is a proxy** for human perception, not a substitute; a human study is the natural
  next step.
- **Scene-content confound.** The emotion/affect tiers prepend a scene template, so high
  recovery on those tiers can partly reflect CLIP recognising the injected scene content
  rather than affect transfer. The *affect-minus-emotion* delta (shared scene, added
  valence/arousal cues) partially isolates the continuous-grounding effect.
- **Probe co-design.** The CLIP recovery prompts share affective vocabulary with the
  conditioning side; we mitigate with a per-emotion template ensemble and report results, but
  an independent affect classifier would strengthen external validity.
- The demo lexicon is small (swap in empirical norms for research runs), and the Ekman
  taxonomy omits mixed affect.

## 10. Reproducibility

`pip install -e ".[dev,research]" && pip install -e ".[ml]"`, then `make benchmark` followed
by `make reproduce` (or `make smoke` for a quick run on the sample). Everything is seeded; the
deterministic core is covered by unit tests in CI.

## References

See `references.bib`.
