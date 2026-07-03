# Runbook: from honest null to a powered result

The committed pilot is a null measured with a degenerate probe. This is the
ordered checklist for turning it into a real measurement on a GPU box. Each
step has an acceptance gate; do not proceed past a failed gate.

## 0. Environment

```bash
uv pip install -r requirements.lock   # the exact paper environment
make test                             # all green, no models needed
make repro-check                      # committed numbers re-derive bit-for-bit
```

## 1. Fix the instrument (blocks everything else)

Measure candidate probes in-domain on EmoSet scenes; the committed default is
CLIP ViT-B/32. Note `--label-key emotion`: EmoSet118K keeps label names in its
`emotion` column, and the loader refuses to sample silently unmappable labels.

```bash
make validate-probe-scene   # ViT-B/32, the committed default
python -m novavision.eval.validate_probe --hf-dataset xodhks/EmoSet118K --label-key emotion \
  --n 400 --split train --seed 0 --clip-model openai/clip-vit-large-patch14 \
  --out results/paper/probe_validation_scene_l14.json
make robustness PROBE_MODEL=<hf-image-emotion-model>   # independent non-CLIP probe
```

**Gate:** the chosen probe must (i) use clearly more than 2 of 7 labels on
*generated* scenes and (ii) beat the majority-class baseline. A probe that
fails this gate makes every downstream recovery number uninterpretable,
including nulls.

Committed evidence (`results/paper/probe_validation*.json`, CPU, 2026-07-03;
macro-F1 averages the six supported classes, since the Ekman mapping leaves
`neutral` unsupported in both sets):

| Probe | EmoSet scenes (n=400) | Faces (n=200) | Labels used in-domain |
|---|---|---|---|
| CLIP ViT-B/32 (committed default) | 40.3% / F1 0.44 | 29.0% / F1 0.22 | 7/7, most frequent `neutral` 28.3% |
| CLIP ViT-L/14 (candidate) | 45.5% / F1 0.47 | 37.5% / F1 0.36 | 7/7, most frequent `sadness` 27.8% |

The L/14 edge is significant per set (exact paired McNemar: EmoSet p=0.038,
faces p=0.040; regenerate with `python scripts/compare_probes.py
results/paper/probe_validation_scene.json results/paper/probe_validation_scene_l14.json`).
Both probes read real scenes; the pilot's neutral-collapse appears only on
generated images, so rerun the pilot under ViT-L/14 (`--probe-model
openai/clip-vit-large-patch14`) and check `probe_health` before committing to
the full n=420 run.

## 2. Powered run

```bash
make reproduce                  # content track: 512 px, 20 subjects, 3 seeds, n=420/tier
make benchmark && make text     # text track on AffectBench
```

Committing a new run changes the locked records: regenerate and commit
`results/paper/results.json`, `make paper` tables, and figures **together**, or
`make repro-check` will fail in CI (by design).

## 3. Human study (needs three or more raters)

```bash
python -m novavision.eval.human_study build --results results/paper --n 60
# one ratings sheet per rater, then:
python -m novavision.eval.human_study analyze --ratings rater1.csv --key results/paper/human_study/key.csv
```

**Gate:** report Cohen's kappa beside every probe number; the realized
per-class counts are printed at build time and written into the study README.
`build_sheet` refuses non-deterministic backends, so the rated images are
always the scored images.

## 4. Publish

- `make paper`, sync the README results table, commit in one change.
- Tag a release so the Zenodo webhook mints a new DOI version.
- Update the paper's pilot framing: the null was the protocol proving itself;
  the powered run is the measurement.
