# Human study protocol

Ground-truths the CLIP recovery probe against human perception. The probe is only
a proxy; this measures how well it agrees with people (Cohen's kappa), which is
the missing leg of the probe-validation story.

## 1. Build the rating sheet

```bash
python -m novavision.eval.human_study build --results results/paper --n 60
```

This regenerates a class-stratified sample of the run's images under
`results/paper/human_study/` (deterministic, so the rated images are byte-identical
to the scored ones) with:

- `images/` — the images to rate,
- `ratings_template.csv` — one blank `emotion` column to fill,
- `key.csv` — the intended and probe labels, **kept hidden from raters**,
- `README.md` — realized per-class counts (report them with any result).

## 2. Recruit and instruct raters

- **Three or more independent raters**, one copy of `ratings_template.csv` each.
- For every image, pick the single best-fitting label from: anger, disgust, fear,
  joy, neutral, sadness, surprise.
- Rate on the image alone; do not show raters the prompt, the intended emotion, or
  each other's sheets.
- Allow "can't tell" only by leaving the cell blank; blanks are excluded, not
  guessed.

## 3. Score agreement

```bash
python -m novavision.eval.human_study analyze --ratings rater1.csv --key results/paper/human_study/key.csv
```

Reports, per rater: `human_vs_probe_kappa` (the headline), plus human-vs-intended
and probe-vs-intended accuracy, and the count of unscored cells. Synonyms (happy →
joy, sad → sadness) are absorbed; anything unmappable is listed, not silently
dropped.

## 4. Report

- Cohen's kappa for each rater and the mean, beside the realized per-class counts.
- Inter-rater agreement (e.g. Fleiss' kappa) as the ceiling: the probe cannot be
  expected to agree with people more than people agree with each other.
- A low human-vs-probe kappa alongside a higher human-vs-intended accuracy is
  direct evidence the probe, not the generator, is the binding limitation.
