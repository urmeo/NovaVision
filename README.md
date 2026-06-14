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
3. **Condition** — the prompt is built at one of three levels (raw, emotion, affect-grounded);
   these tiers are the ablation.
4. **Generate** — Stable Diffusion Turbo renders the image from a fixed seed.
5. **Recover** — CLIP (ViT-B/32) reads the emotion back from the image and compares it to the
   intended label.

## Evaluation

<p align="center">
  <img src="screenshots/emotion_analysis.png" alt="Emotion analysis" width="780">
</p>

The benchmark is built from GoEmotions under the official Ekman mapping, balanced across the
seven classes. Each item is conditioned on its gold emotion — isolating controllability from
upstream classifier error — and scored on recovery accuracy, macro-F1, valence/arousal
correlation (Pearson r), and CLIP-T. Confusion matrices and valence/arousal plots are written
to `results/`. The method is described in the paper, `paper/paper.md`.

## Tech stack

| Area | Tools |
|------|-------|
| ML / NLP | PyTorch, Hugging Face Transformers, Diffusers (SD-Turbo), CLIP |
| Application | Python, Flask, Gradio |
| Tooling | pytest, ruff, GitHub Actions, Docker |

## Future scope

- [ ] Human study validating CLIP recovery against perceived emotion
- [ ] Larger benchmark beyond the balanced GoEmotions sample
- [ ] Independent affect classifier to remove probe/conditioning overlap
- [ ] Mixed and compound emotions beyond the Ekman set

## License

[MIT](LICENSE)
