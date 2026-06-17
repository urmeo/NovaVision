# Third-party licenses

This repository's [MIT LICENSE](LICENSE) covers **the NovaVision source code only**.
It does **not** relicense the model weights, datasets, or affect norms the code can
download or call — those keep their own terms, several of which are **non-commercial**.
Confirm each before any commercial use.

## Model weights (downloaded at runtime, not redistributed here)

| Model | Used as | License | Note |
|---|---|---|---|
| `stabilityai/sd-turbo` | default image generator (`diffusers`, `hf-api`) | Stability AI **Community License** | Free for research/personal; **commercial use requires a Stability AI membership**. SD-Turbo is distilled from Stable Diffusion 2.1. |
| `openai/clip-vit-base-patch32` | recovery probe (`CLIPProbe`) | MIT | Permissive. |
| `j-hartmann/emotion-english-distilroberta-base` | text emotion classifier | see model card (base DistilRoBERTa is Apache-2.0) | Check the model card for the fine-tuned checkpoint's terms. |

The hosted `hf-api` backend can target any HF text-to-image model; the model you point
it at carries its own license (e.g. `black-forest-labs/FLUX.1-dev` is non-commercial — it
is **not** a default here).

## Datasets (downloaded by the user, not redistributed here)

| Dataset | Used for | License |
|---|---|---|
| GoEmotions (`google-research-datasets/go_emotions`) | building AffectBench (text) | Apache-2.0 |
| EmoSet / FastJobs Visual_Emotional_Analysis | probe validation (images) | each dataset's own terms (research use) |

## Affect lexicon

- **Shipped file — `data/lexicon/affect_lexicon.tsv`:** a small **original demo lexicon**
  authored for NovaVision, covered by this repo's MIT license. It is for offline use and
  tests, not a research-grade resource.
- **Warriner et al. (2013) norms** (`scripts/download_lexicon.py`): fetched by the user
  from the original source (CRR, Ghent University); **not redistributed** in this repo.
  Their academic terms apply.
- **NRC-VAD:** *not shipped and not downloaded by this project.* If you supply your own
  NRC-VAD TSV via `NOVAVISION_LEXICON`, note NRC-VAD is **free for research but requires a
  license for commercial use** (v2 figures are CC-BY-NC-SA 4.0). Do not redistribute it.

## Summary

- **Code:** MIT — reuse freely.
- **Weights/data/norms:** keep their own terms. For **commercial** use, the binding
  constraints are SD-Turbo (Stability membership) and any non-commercial lexicon you add
  (NRC-VAD). Swap SD-Turbo for a permissively-licensed generator and use a CC-BY/Apache
  VAD source if you need an unencumbered commercial stack.
