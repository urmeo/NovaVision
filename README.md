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

- The **content** track renders neutral content (`data/content_bank.txt`) under each intended
  emotion, so the score reflects conditioning rather than scene content.
- The **text** track (`--track text`) conditions on **AffectBench** — a GoEmotions-derived
  benchmark (test split, deduplicated, balance-checked, datasheeted in `data/DATASHEET.md`) —
  with text-grounded valence/arousal and a shuffled-emotion floor.
- Conditions: `raw` (control), `naive` (bare emotion word), `emotion` (engineered modifier),
  `affect` (+ valence/arousal), plus a floor (`scene` or `shuffled`).
- Scored on recovery accuracy (with bootstrap 95% CIs), macro-F1, valence/arousal correlation
  (Pearson r and Spearman ρ), and CLIP-T; tier deltas get a paired significance test.
- The recovery probe is **validated** against a labelled set (`make validate-probe`): CLIP
  ViT-B/32 zero-shot recovers emotion at only **29% accuracy** — a weak instrument that bounds
  every recovery score. An independent non-CLIP probe (`--probe hf`) and a human study
  (`novavision.eval.human_study`, Cohen's κ) are wired in.
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

- [ ] A stronger recovery probe — the CLIP ViT-B/32 zero-shot probe is weak (29% accuracy)
- [ ] Scale the human study (3+ raters) and report Cohen's κ against the probe
- [ ] Scene-affect probe validation on EmoSet (the harness already supports it)
- [ ] Image-grounded conditioning (EmoSet/FI) for the image-affect claim
- [ ] Mixed and compound emotions beyond the Ekman set

## License

[MIT](LICENSE)
