# Provenance

Where every model, weight, and eval prompt in NovaVision comes from, pinned to
the exact version used for the committed results. This complements
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (the license terms) and
`data/DATASHEET.md` (the benchmark data card).

## Trained vs. off-the-shelf

**NovaVision trains nothing.** Every model is a pre-trained, off-the-shelf
checkpoint loaded by `from_pretrained` at a pinned commit; no weights in this
repo are fine-tuned, distilled, or otherwise learned by this project. The only
components authored here are:

- the deterministic prompt-templating that layers emotion as a modifier over
  independent content (`novavision/prompting.py`), and
- the pure-numpy evaluation statistics (`novavision/eval/metrics.py`).

Neither is a learned model, so there are no NovaVision training data, no
NovaVision-trained weights, and nothing to release beyond source. This matters
for the honesty of the benchmark: the recovery signal (or its absence) is a
property of off-the-shelf SD-Turbo + off-the-shelf CLIP under a fixed protocol,
not of anything tuned to make the number look good.

## Model weights

All three are pinned to an exact Hugging Face commit in
[`novavision/config.py`](novavision/config.py), so a rerun pulls identical
weights. Revisions are also written into every run's manifest
(`results/paper/results.json` → `manifest.model_revisions`).

| Role | Model (HF id) | Pinned revision | Base / lineage | License |
|---|---|---|---|---|
| Text-to-image generator | `stabilityai/sd-turbo` | `b261bac6fd2cf515557d5d0707481eafa0485ec2` | SD-Turbo, adversarial-diffusion-distilled from Stable Diffusion 2.1 | Stability AI **Community License**: free for research/personal use; **commercial use requires a Stability AI membership** |
| Recovery probe | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | CLIP ViT-B/32 | MIT |
| Text emotion classifier | `j-hartmann/emotion-english-distilroberta-base` | `0e1cd914e3d46199ed785853e12b57304e04178b` | fine-tuned DistilRoBERTa (base is Apache-2.0) | see the model card for the fine-tuned checkpoint's terms |

Notes:

- **Not redistributed.** These weights are downloaded at runtime from Hugging
  Face; this repo ships none of them.
- **Generator is swappable.** `--diffusion-model` (and the `hf-api` backend)
  can point at any HF text-to-image model; SD-Turbo is the only default. Any
  model you substitute carries its own license (e.g. `FLUX.1-dev` is
  non-commercial and is **not** a default here).
- **Probe is swappable.** The default probe is CLIP ViT-B/32; `--probe hf`
  selects an independent HF image-emotion classifier through the same
  interface, so the recovery instrument can be changed without touching the
  protocol.

## Emotion classifier vs. recovery probe

Two distinct off-the-shelf models sit at the two ends of the pipeline, and they
are deliberately different so recovery is not the classifier grading itself:

- **Detect (input side):** the DistilRoBERTa classifier scores the six Ekman
  emotions plus neutral in the *input text* (`novavision/affect/analyzer.py`).
  On the **content** track this is not used to score recovery (the intended
  label is assigned by construction); it is exercised on the **text** track,
  where its prediction is logged separately as classification accuracy.
- **Recover (image side):** the CLIP probe reads emotion and valence/arousal
  back from the *generated image* (`novavision/eval/probes.py`). This is the
  measurement instrument; §7 of the paper and `make validate-probe*` quantify
  its known errors (29.0% on out-of-domain faces; in-domain collapse onto
  `neutral`).

## Affect grounding (valence/arousal)

Valence/arousal are not a model output; they are looked up from an affect
lexicon and blended with the emotion's circumplex prior by lexical coverage
(`novavision/affect/lexicon.py`). Lexicon provenance:

- **Shipped (`data/lexicon/affect_lexicon.tsv`):** a small **original demo
  lexicon** authored for NovaVision (MIT, this repo). It is for offline use and
  tests, not a research-grade norm set.
- **Warriner et al. (2013)** norms are fetched by the user from the original
  source via `scripts/download_lexicon.py`; **not redistributed** here.
- **NRC-VAD** is neither shipped nor downloaded; if supplied via
  `NOVAVISION_LEXICON` it keeps its own (non-commercial) terms.

## Eval prompt sets

There are two, built by frozen, documented procedures, never hand-picked to
move a number.

### Content track: the 20-subject content bank

- **Source:** `data/content_bank.txt`, a fixed list of twenty **affect-neutral**
  content subjects (e.g. "a city street with buildings").
- **How constructed:** authored once so that the depicted subject is independent
  of the intended emotion. Emotion enters only as a prompt modifier over the
  tiers (`raw → naive → emotion → affect`) in `novavision/prompting.py`; the
  `scene` floor uses fixed per-emotion scenes and is a control, not a content
  choice. Because content ⟂ emotion by construction, any recovery signal must
  come through the modifier, not the scene.
- **Committed pilot:** the first 2 subjects × 7 emotions × 1 seed → n=14 per
  conditioning tier (n=7 for `scene`).

### Text track: AffectBench

- **Source:** derived on demand from **GoEmotions** (Demszky et al., 2020),
  `simplified` config, GoEmotions revision pinned, license Apache-2.0.
- **How constructed** (`novavision/data/build_benchmark.py`; full card in
  `data/DATASHEET.md`): single-label examples only, mapped to the six Ekman
  emotions plus neutral via the dataset's official grouping
  (`novavision/taxonomy.py:GOEMOTIONS_TO_EKMAN`); sampled from the **test**
  split; exact within-sample dedup after normalisation; **cross-split dedup**
  subtracts any item whose normalised text also appears in the **train** split
  (so no eval sentence is one a model could have trained on); deterministic
  seeded shuffle, round-robin interleave to keep any prefix balanced.
- **Manifest:** the build records the pinned revision, seed, realized per-class
  counts, `dropped_train_overlap`, `total`, a `balanced` flag, and a content
  hash. Report the realized per-class n with any result; under the Ekman
  collapse the scarce classes (notably `disgust`) underfill.
- **Not a reported source:** the hand-authored
  `tests/fixtures/affectbench_sample.csv` is a test fixture only and can never
  produce a reported number (`load_benchmark` has no default path).

## Run provenance

Every run writes a manifest to `results/paper/results.json` recording the git
SHA, Python version, platform, library versions (torch, transformers,
diffusers, numpy, datasets, pillow), model revisions, device, dtype, config,
and, for the text track, the benchmark SHA-256. `requirements.lock` pins the
exact resolved environment. Tables and figures are regenerated from the
committed records by `scripts/report.py` / `make paper`, never hand-written, and
`make repro-check` re-derives the headline numbers from the raw records to guard
against drift.
