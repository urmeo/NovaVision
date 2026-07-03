# AffectBench: datasheet

A short data card in the spirit of Gebru et al. (2021), "Datasheets for Datasets".

## Motivation

AffectBench is the text benchmark for NovaVision's secondary, text-conditioned
track: each item is a sentence with one Ekman emotion, used to condition image
generation and then to test whether the intended emotion is recovered from the
image. It exists to make that evaluation reproducible from a public source
rather than from hand-written examples.

## Composition

- **Instances.** Short English sentences, each labelled with one of the six Ekman
  emotions plus neutral (seven classes): anger, disgust, fear, joy, sadness,
  surprise, and neutral.
- **Source.** Derived from GoEmotions (Demszky et al., 2020), `simplified`
  config, using only single-label examples mapped to Ekman via the dataset's
  official grouping (`novavision/taxonomy.py:GOEMOTIONS_TO_EKMAN`).
- **Split.** Sampled from the GoEmotions **test** split by default, so items do
  not overlap a model's training split.
- **Size / balance.** `n` items per class (configurable). The build records the
  realized per-class counts in `affectbench.manifest.json`; if a class underfills
  it is logged and the set is flagged `balanced: false`. Under the Ekman collapse,
  `disgust` is the scarcest class.

## Collection & preprocessing

- Built on demand by `python -m novavision.data.build_benchmark`.
- Exact duplicates are removed after whitespace/case normalisation.
- **Cross-split dedup:** any item whose normalised text also appears in the
  GoEmotions **train** split is subtracted (default; disable with
  `--keep-train-overlap`), so no eval sentence is one a model could have trained
  on. The number removed is logged as `dropped_train_overlap`.
- The pool is sorted before a seeded shuffle (deterministic), then interleaved
  round-robin across classes so any prefix stays balanced.
- The GoEmotions revision is pinned; the manifest stores split, revision, seed,
  realized per-class `counts`, `total`, `dropped_train_overlap`, and a `balanced`
  flag. **Report the realized per-class `n`** with any result. Under the Ekman
  collapse the scarce classes (notably `disgust`) underfill.
- No tuning touches eval items: the recovery-probe templates, the affect lexicon,
  and the content bank are authored independently and frozen before measurement.

## Uses & limitations

- GoEmotions is **text** emotion (Reddit comments), not image affect. Results on
  AffectBench are therefore scoped to *text-prompt* controllability; for image
  affect, validate the probe against an image-grounded set (see the paper).
- **Domain bias.** GoEmotions is Reddit-English with demographic skew and modest
  Ekman-level inter-annotator agreement ($\kappa \approx 0.33$ to $0.44$).
  AffectBench inherits this; it is not a balanced sample of human affect, and
  rare-class results are underpowered.
- The hand-authored `tests/fixtures/affectbench_sample.csv` is a **test fixture
  only** and must never produce a reported number.

## Distribution & licensing

GoEmotions is released under Apache-2.0. AffectBench redistributes derived text
under the same terms; cite Demszky et al. (2020). Project code is MIT.
