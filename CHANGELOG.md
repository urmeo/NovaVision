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

## [1.0.0] - 2026-07-02

- First public release: the reproducible emotion-controllability protocol, the
  committed CPU-pilot honest null, AffectBench, and the full evaluation harness.
