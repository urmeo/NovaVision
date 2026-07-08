# Model card: NovaVision

Following Mitchell et al. (2019), "Model Cards for Model Reporting". This documents
the **system** (the emotion-to-image pipeline and its evaluation harness); the
AffectBench **data** card is `data/DATASHEET.md`. The default models are pinned to
exact revisions in `novavision/config.py`; each run's manifest records the
revisions actually used (a swapped model loads unpinned).

## Overview

NovaVision conditions an image on the emotion of a sentence, then recovers the
emotion from the image with a swappable probe and compares it to the intent. It is
a research protocol and evaluation harness, not a product. It **trains nothing**:
every model is an off-the-shelf, pinned checkpoint.

| Component | Model | License |
|---|---|---|
| Emotion detection | `j-hartmann/emotion-english-distilroberta-base` | see model card |
| Image generation | `stabilityai/sd-turbo` | Stability AI Community (non-commercial without membership) |
| Recovery probe | `openai/clip-vit-base-patch32` | MIT |

## Intended use

- Research on emotion controllability in text-to-image generation.
- Reproducing and extending the protocol; scoring new generators/probes under it.
- A local demo of the pipeline (the web app).

## Out-of-scope and prohibited use

- **Not an emotion-recognition system.** Do not use it to infer, score, or act on
  any real person's emotions. It reads the affect of *text* and the affect a probe
  reads from *generated art*, neither of which is a person's felt emotion.
- **Not a deployed service.** Single generator, single probe, research pilot.
- No high-stakes, clinical, surveillance, employment, or safety decisions.

## Factors and limitations

- **The probe is the binding limitation.** CLIP ViT-B/32 recovers emotion at 40.3%
  on real scenes and *collapses onto `neutral`* on the pilot's generated scenes; no
  recovery number is interpretable as controllability until a probe clears the
  in-domain gate (`make validate-probe-scene`). Every number is reported with its measured error and,
  via Rogan-Gladen, corrected for it.
- **Labels are coarse and culturally situated.** Six Ekman emotions plus neutral;
  mixed and compound affect are out of scope.
- **Taxonomy mismatch with image-emotion evaluators.** Standard evaluators are
  trained on Mikels' eight categories (no neutral); the Ekman-plus-neutral target
  space requires an explicit mapping (see the paper's Limitations).
- **Domain skew.** The text track derives from GoEmotions (Reddit-English, modest
  inter-annotator agreement); AffectBench inherits that bias.

## Evaluation and ethics

- The committed result is an honest, guarded null from a calibration pilot (n=14
  per tier), read as a calibration of the instrument rather than a controllability
  score; the *properly powered* run (n=420 per tier, preregistered) has not been
  executed yet (see `paper/paper.pdf`, `PREREGISTRATION.md`). The committed numbers
  are re-derived from the records in CI (`make repro-check`).
- Generated images can carry the biases of SD-Turbo; the pipeline applies no
  content filtering and must not be exposed publicly without one.
- MIT covers the code only; model weights and data keep their own upstream terms
  (SD-Turbo is non-commercial), stated on each model's Hugging Face page.
