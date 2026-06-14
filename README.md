# NovaVision

Text-to-image generation conditioned on the emotion of a sentence — with a benchmark that
measures whether the generated image actually conveys that emotion.

<p align="center">
  <img src="screenshots/main_interface.png" alt="NovaVision interface" width="860">
</p>

Most emotional image generators are judged by eye. NovaVision treats emotion as something to
verify: it detects the emotion in text, grounds it in continuous valence and arousal,
conditions the prompt, generates an image, then recovers the emotion from that image with CLIP
and checks it against the original intent.

<p align="center">
  <img src="screenshots/demo_preview.gif" alt="Demo" width="860">
</p>

## How it works

<p align="center">
  <img src="screenshots/how_it_works.png" alt="Pipeline" width="780">
</p>

- **Detect** — a DistilRoBERTa classifier scores seven Ekman emotions.
- **Ground** — valence and arousal are computed from an affect lexicon, blended with the
  emotion's circumplex prior in proportion to how many affect words the text contains.
- **Condition** — the prompt is built at one of three levels (raw, emotion, affect-grounded),
  which also serves as the ablation.
- **Recover** — CLIP reads the emotion back from the generated image; its agreement with the
  intended label is the core metric.

## Highlights

- Emotion is measured from the text, not mapped from fixed per-class constants.
- Evaluation is automatic and reproducible (affect recovery), not manual inspection.
- Image backends are pluggable: local Stable Diffusion Turbo (seedable), the hosted API, or an
  offline backend for tests.
- Seeded end to end; the deterministic core is covered by tests and CI.

## Evaluation

<p align="center">
  <img src="screenshots/emotion_analysis.png" alt="Emotion analysis" width="780">
</p>

The benchmark builds a balanced, emotion-labelled set from GoEmotions under the Ekman grouping
and scores the three conditioning tiers on recovery accuracy, macro-F1, valence/arousal
correlation, and CLIP-T. Confusion matrices and valence/arousal plots are generated into
`results/`. Method and design are written up in `paper/paper.md`.

## Tech stack

- **ML / NLP** — PyTorch, Hugging Face Transformers, Diffusers (SD-Turbo), CLIP
- **Application** — Python, Flask, Gradio
- **Tooling** — pytest, ruff, GitHub Actions, Docker

## Project structure

```
novavision/   core library: taxonomy, affect, prompting, generation, eval, pipeline
data/         sample benchmark and demo lexicon
paper/        method write-up and references
tests/        test suite
server.py     web API and frontend
app.py        Gradio interface
```

## Quickstart

```bash
git clone https://github.com/urme-b/NovaVision.git
cd NovaVision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,research]" && pip install -e ".[ml,app]"
python server.py
```

## Citation

Cite using the metadata in `CITATION.cff`.

## License

MIT — see `LICENSE`.
