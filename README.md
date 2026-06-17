# NovaVision

Text-to-image generation conditioned on the emotion of a sentence — with a **reproducible
protocol and evaluation harness** that measures whether the generated image actually conveys that
emotion, designed around the failure modes that make such a measurement easy to fake.

<p align="center">
  <img src="screenshots/main_interface.png" alt="NovaVision interface" width="860">
</p>

> **What this is — and isn't.** This is a *protocol* and *harness* for emotion controllability,
> plus **AffectBench**, a frozen GoEmotions-derived **text** benchmark. It is **not** a
> system-ranking benchmark: the shipped runs cover one generator (SD-Turbo), one style, and one
> recovery probe (CLIP ViT-B/32), and the committed result is a small CPU **pilot** reporting an
> honest **null** (see [Results](#evaluation)). The harness is built to compare generators,
> styles, and probes (`--diffusion-model`, `--style`, `--probe`), but those comparisons are a
> capability, not a delivered result. Probe-bounded: CLIP recovery is a weak instrument (see
> [Probe validation](#evaluation)), so every recovery number is read against its measured error.

## Getting started

```bash
git clone https://github.com/urme-b/NovaVision && cd NovaVision
make setup          # core + dev deps (deterministic core, tests, lint)
make test           # 73 tests, no models needed (runs in seconds)

# To run the real models (downloads SD-Turbo, CLIP, the emotion classifier):
make setup-ml
make app            # launch the web app at http://127.0.0.1:8000
make smoke          # quick end-to-end benchmark run (2 subjects, 1 seed)

# Reproduce the paper artifacts:
make reproduce      # canonical content-track run -> results/paper/
make validate-probe-scene   # in-domain probe ceiling on EmoSet
make paper          # regenerate the paper tables/figures from results
```

For the exact paper environment, install the pinned lockfile instead:
`uv pip install -r requirements.lock`.

## Why it matters

Most emotional image generators are judged by eye. NovaVision treats emotion as something to
verify: it detects the emotion in text, grounds it in continuous valence and arousal,
conditions the prompt, generates an image, then recovers the emotion from that image with a
swappable probe and compares it to the original intent. The result is an automatic, reproducible
*protocol* for emotional controllability — bounded by a no-emotion control, a template ceiling,
and a validated probe, so a number means something rather than flattering the method.

<p align="center">
  <img src="screenshots/demo_preview.gif" alt="Demo" width="860">
</p>

## How it works

<p align="center">
  <img src="screenshots/how_it_works.png" alt="Pipeline" width="780">
</p>

1. **Detect** — DistilRoBERTa scores the seven Ekman emotions in the text.
2. **Ground** — valence and arousal are estimated from an affect lexicon, blended with the
   emotion prior by word coverage: `v = c·v_lex + (1−c)·v_prior`.
3. **Condition** — image content stays independent of the emotion; emotion enters only as a
   modifier over three tiers (raw, emotion, affect). The tiers are the ablation, so recovery is
   attributable to the conditioning, not a canned scene.
4. **Generate** — Stable Diffusion Turbo renders the image from a fixed seed.
5. **Recover** — a swappable probe (CLIP ViT-B/32 by default) reads the emotion and graded
   valence/arousal back from the image and compares to the intended label.

The claim is bounded on both sides. **raw** is the negative control (no emotion → chance);
**scene** is the positive control and template ceiling (a fixed per-emotion scene a working probe
should read easily). Because a probe that has *collapsed* onto one label also scores at chance,
every run reports a **majority-class baseline** and a **probe-collapse** diagnostic next to the
numbers — recovery only counts when it clears the baseline with a non-degenerate probe. Tier
differences come with bootstrap confidence intervals and a paired significance test.

## Evaluation

<p align="center">
  <img src="screenshots/emotion_analysis.png" alt="Emotion analysis" width="780">
</p>

- The **content** track renders neutral content (`data/content_bank.txt`) under each intended
  emotion, so the score reflects conditioning rather than scene content.
- The **text** track (`--track text`) conditions on **AffectBench** — a GoEmotions-derived
  benchmark (test split, deduplicated, balance-checked, datasheeted in `data/DATASHEET.md`) —
  with text-grounded valence/arousal and a shuffled-emotion floor.
- Conditions: `raw` (control), `naive` (bare emotion word), `emotion` (engineered modifier),
  `affect` (+ valence/arousal), plus a floor (`scene` or `shuffled`).
- Scored on recovery accuracy (with bootstrap 95% CIs), macro-F1, valence/arousal correlation
  (Pearson r and Spearman ρ), and CLIP-T; tier deltas get a paired significance test.
- The recovery probe is **validated** and treated as a known-error instrument, not assumed:
  - **Out-of-domain (faces, `make validate-probe`):** CLIP ViT-B/32 zero-shot recovers the Ekman
    emotion at only **29% accuracy** on facial-expression photos.
  - **In-domain (generated scenes):** every benchmark run emits a `probe_health` diagnostic, and
    on the committed pilot the probe **collapses onto `neutral` (90% of images, 2 of 7 labels
    used)** — so the headline numbers are read against a probe that barely discriminates, *in the
    domain it is actually used*. The 29% faces figure is an out-of-domain proxy, not the operating
    error.
  - `make validate-probe-scene` runs the same check in-domain on EmoSet; an independent non-CLIP
    probe (`--probe hf`, `make robustness`) and a human study (`novavision.eval.human_study`,
    Cohen's κ) slot into the same interface to cross-check the headline.
- Confusion matrices and valence/arousal plots are written to `results/`; `make paper`
  regenerates the paper tables from `results/paper/results.json`.
- Method details in the paper, `paper/paper.md`.

## Tech stack

| Area | Tools |
|------|-------|
| ML / NLP | PyTorch, Hugging Face Transformers, Diffusers (SD-Turbo), CLIP |
| Application | Python, Flask, Gradio |
| Tooling | pytest, ruff, GitHub Actions, Docker |

## Future scope

The honest blocker is the instrument: the recovery probe is too weak to measure in-domain affect,
so a powered run is only worth it once the probe is fixed. In priority order:

- [ ] **A recovery probe that actually reads generated scenes.** CLIP ViT-B/32 collapses onto
      `neutral` in-domain; validate a stronger/independent probe (`--probe hf`, ViT-L/14) on
      EmoSet (`make validate-probe-scene`) *before* trusting any recovery number.
- [ ] **The powered run** (`make reproduce`: 512-px, 20 subjects, 3 seeds → n=420) on a GPU box,
      once a non-degenerate probe clears the in-domain ceiling.
- [ ] Scale the human study (3+ raters) and report Cohen's κ against the probe.
- [ ] Cross-system comparison (multiple generators/styles/probes) to make this a ranking benchmark.
- [ ] Mixed and compound emotions beyond the Ekman set.

## License

[MIT](LICENSE)
