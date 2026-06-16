# NovaVision

Text-to-image generation conditioned on the emotion of a sentence — with a benchmark that
measures whether the generated image actually conveys that emotion.

<p align="center">
  <img src="screenshots/main_interface.png" alt="NovaVision interface" width="860">
</p>

## Why it matters

Most emotional image generators are judged by eye. NovaVision treats emotion as something to
verify: it detects the emotion in text, grounds it in continuous valence and arousal,
conditions the prompt, generates an image, then recovers the emotion from that image with CLIP
and compares it to the original intent. The result is an automatic, reproducible measure of
emotional controllability rather than a subjective one.

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

The claim is bounded by two floors a tautology cannot beat: a **no-emotion control** (recovery
should sit at chance) and a **template-only** ceiling (fixed scene, no content). Tier differences
are reported with bootstrap confidence intervals and a paired significance test.

## Evaluation

<p align="center">
  <img src="screenshots/emotion_analysis.png" alt="Emotion analysis" width="780">
</p>

- **AffectBench** is built from GoEmotions under the official Ekman mapping — sampled from the
  test split, deduplicated, balance-checked, and shipped with a datasheet (`data/DATASHEET.md`).
- The primary track renders neutral content under each intended emotion, so the score reflects
  conditioning rather than scene content.
- Scored on recovery accuracy (with bootstrap 95% CIs), macro-F1, valence/arousal correlation
  (Pearson r and Spearman ρ), and CLIP-T; tier deltas get a paired significance test.
- A probe-validation and human-study harness (`novavision.eval.human_study`) checks the CLIP
  proxy against people (Cohen's κ).
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

- [ ] Scale the human study (3+ raters) and report Cohen's κ against the probe
- [ ] Independently-trained image-emotion probe to fully remove probe/conditioning overlap
- [ ] Image-grounded conditioning (EmoSet/FI) alongside the GoEmotions text track
- [ ] Mixed and compound emotions beyond the Ekman set

## License

[MIT](LICENSE)
