# Changelog

All notable changes to NovaVision. Format follows Keep a Changelog; versions
follow the git tags.

## [Unreleased]

- Recovery correction for probe error (Rogan-Gladen), Holm family-wise correction,
  and Cohen's h effect sizes in the generated tables.
- Sample-size / power analysis, run checkpoint-and-resume, affect-blend ablation.
- Public API (`novavision.build_pipeline`, `run_experiment`), submission schema,
  Colab reproduction notebook, preregistration, and system model card.
- Content-Security-Policy and type guards in the web UI; `py.typed` marker.
- Repositioned related work against the 2024-2026 emotion-conditioned generation
  literature (EmoGen, EmotiCrafter, EmoEdit, MUSE, EPIG); taxonomy-mismatch analysis.
- Replaced the injected-markdown paper with a LaTeX preprint (paper/paper.tex,
  compiled paper/paper.pdf); tables render to paper/tables.md as before.
- Removed the Gradio/Spaces path (`app.py`, `spaces_config.yaml`, `requirements.txt`):
  `server.py` is the single entry point, served by gunicorn in Docker. Removed
  `RUNBOOK.md`, `PROVENANCE.md`, and `THIRD_PARTY_LICENSES.md`; model pins and per-run
  manifests carry the provenance, upstream pages carry the license terms.
- Removed `requirements.lock`; Docker and CI install from pyproject extras, and the
  per-run manifest is the environment record.

## [1.0.0] - 2026-07-02

- First public release: the reproducible emotion-controllability protocol, the
  committed CPU-pilot honest null, AffectBench, and the full evaluation harness.
