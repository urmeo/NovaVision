# A Reproducible Protocol for Measuring Emotion Controllability in Text-to-Image Generation

**Urme Bose** (`urme-b`)

> Working draft. The Results tables and figures are produced by `scripts/report.py`
> from the run in `results/paper/` (`make paper`), never hand-written. The committed
> numbers are a CPU **pilot** (`make pilot`); `make reproduce` is the powered
> configuration. The pilot's purpose is to show the protocol and its diagnostics run
> end-to-end and that the recovery probe is currently too weak to support any positive
> claim — not to report a controllability result.

## Abstract

Emotional text-to-image generation is usually judged by inspection, which conflates the emotion
*intended* by the prompt with the emotion an image actually conveys. A natural automatic
alternative — condition on an emotion, then recover it from the image with a zero-shot probe —
is easy to get wrong: if the conditioning paints a fixed per-emotion scene and the probe is
written to match it, "recovery" becomes a tautology. We make the protocol honest. Image
*content* is held independent of the intended emotion, emotion enters only as a modifier, and
the headline number is bounded by two floors a tautology cannot beat: a no-emotion control
(chance) and a template-only ceiling (pure scene recognition). We report every tier with
bootstrap confidence intervals and a paired significance test, and release **AffectBench**, a
frozen, datasheeted GoEmotions-derived benchmark, for a text-conditioned track. The contribution
is the protocol and artifact — designed around its failure modes — not a single controllability
score. We also validate the recovery probe and find that CLIP zero-shot is a **weak instrument**:
29% accuracy out of domain (on faces) and, worse, *degenerate in domain* — on the generated
scenes it collapses onto `neutral` (2 of 7 labels, 90% of items). Every recovery number, including
our reported null, is read against that ceiling, and we withhold the powered run until the probe
can see more than two of seven emotions in the domain it is used.

## 1. Introduction

Given an intended emotion, does a conditioned generator produce an image whose emotion a probe
recovers as the intended one — and does that signal come from the *method* rather than from a
canned scene or a probe written to agree with the prompt? We answer with an automatic protocol
designed around its own failure modes, rather than a single accuracy number taken at face value.

Our claim is deliberately modest and falsifiable: conditioning must beat a no-emotion control,
and must add information beyond a fixed template, with the difference significant under a paired
test. If it does not, we report the null — a rigorous null with a bounded instrument is more
useful than a circular positive.

## 2. Related work

Affect is represented with the circumplex model of valence and arousal [@russell1980]. Text
emotion is read with a DistilRoBERTa classifier [@hartmann2022] and lexical affect with
empirical norms [@warriner2013; @mohammad2018]. Image-level emotion has its own datasets and
classifiers — FI [@you2016building], affective classification from psychology/art features
[@machajdik2010], EMOTIC [@kosti2017emotion], WEBEmo [@panda2018contemplating], ArtEmis [@achlioptas2021artemis], and the
large-scale EmoSet [@yang2023emoset] — which we use to *validate the probe* rather than to condition.
Generation uses latent diffusion [@rombach2022; @sauer2023]; emotion-conditioned diffusion such
as EmoGen [@yang2024emogen], EmotiCrafter [@yang2024emoticrafter], and CoEmoGen [@li2025coemogen]
motivates the task and supplies the natural baselines for cross-system comparison under our
protocol. Evaluation uses CLIP zero-shot transfer
[@radford2021], whose use as an affect probe we treat as an instrument to be validated, not
assumed. The benchmark follows datasheet practice [@gebru2021datasheets].

## 3. Method

**Decoupled conditioning.** For content $x$ (a neutral subject, or a benchmark sentence) and an
intended emotion $e$, the prompt is assembled so that $x$ supplies the depicted content and $e$
supplies only a mood/affect modifier. Content is never selected by $e$; the same content is
rendered under every emotion. Conditioning increases over three tiers, which form the ablation:

- **raw** — content and style only (no emotion); the *control*.
- **emotion** — adds a mood modifier for $e$.
- **affect** — also injects valence/arousal cues (palette, lighting), grounded in continuous
  affect ($v = c\,v_{lex} + (1-c)\,v_{prior}$, lexicon-weighted and blended with the circumplex
  prior by coverage $c$). This blend is a **heuristic**, not a fitted model: $c$ is the lexical
  coverage of the input, and the convex combination is not yet validated against held-out VAD
  norms. Its sensitivity is ablatable by forcing $c=0$ (prior only) or $c=1$ (lexicon only); we
  flag this as an open validity question, not a tuned result.

**Floors and controls.** Two conditions bound the claim. **raw** is the *negative* control (no
emotion → recovery should sit at chance, $1/7$). **scene** renders a fixed per-emotion template
with *no* content; it measures pure template recognition and doubles as a *positive* control — a
probe that can read any emotion at all should score highest here, so `scene` $>$ `raw` is the
sanity check that the instrument is not blind. A method is only credible if it clears the
negative control and adds signal over the template. Crucially, "raw at chance" is *not* on its
own evidence the floors discriminate: a probe collapsed onto a single label also scores chance on
balanced data. We therefore report, beside every tier, the **majority-class baseline** and a
**probe-collapse** diagnostic (how many of the seven labels the probe ever uses); recovery counts
only when it clears the baseline with a non-degenerate probe.

**Recovery probe.** A swappable `Probe` reads emotion and graded valence/arousal back from the
image. The default is CLIP zero-shot with two fixes over the naive version: logits are scaled by
the model's learned temperature before softmax (so valence/arousal are not squeezed toward zero),
and valence/arousal are read as the expected value over an *ordered ladder* of anchor prompts
rather than a single positive/negative contrast. Probe templates are frozen before measurement
and are never tuned on evaluation items.

**Circularity is the central threat, and we attack it three ways.** Because the conditioning
writes an emotion word into the prompt and a CLIP-family probe reads an emotion back out, a high
score could measure prompt/probe lexical agreement rather than image content. We do not assume
this away; we bound it. (i) *Decoupled content* means the depicted subject is never chosen by the
emotion, so any signal must come through the mood/affect modifier, not the scene. (ii) An
**independent, non-CLIP image-emotion classifier** (`HFImageClassifierProbe`, `--probe hf`, e.g.
an EmoSet/WEBEmo-style model) slots into the same interface, removing the shared vocabulary
entirely; validating the headline under this probe is a prerequisite, not an option, and the
default CLIP probe is reported only with its measured error (§7). (iii) A **shuffled-label
control** quantifies the circularity baseline directly: we permute the intended labels and
recompute recovery to build the null distribution under *random* targets, and report a one-sided
permutation p-value per tier (§5–6). Recovery is evidence only when it clears this shuffled-label
null with a non-degenerate probe.

## 4. AffectBench

The primary track conditions on a bank of twenty affect-neutral content subjects
(`data/content_bank.txt`), so content is independent of emotion by construction. A secondary
**text** track (§5) conditions on **AffectBench**, where valence/arousal are grounded in the
sentence rather than the emotion prior. AffectBench is built on demand from GoEmotions
[@demszky2020] under the official Ekman grouping — single-label examples mapped to the six Ekman
emotions plus neutral (seven classes), sampled per class from the **test** split, and
interleaved so any prefix stays
balanced. Two dedup passes protect hygiene: exact duplicates are removed *within* the sample, and
any item whose normalised text also appears in the **train** split is subtracted (`build_benchmark`
default; the dropped count is logged), so no benchmark sentence is one a model could have trained
on. The build records realized per-class counts, the train-overlap count, and a content hash; the
probe templates, lexicon, and content bank are frozen and never tuned on these items. A datasheet
(`data/DATASHEET.md`) documents composition, splits, balance, and licensing, and `load_benchmark`
is the loader (no default path, so a run never silently uses a sample). A hand-authored sample
ships only as a test fixture and can never produce a reported number. GoEmotions is Reddit-English
with modest inter-annotator agreement (Ekman-level $\kappa \approx 0.33$–$0.44$), so AffectBench
inherits that domain skew; results are scoped to *text-prompt* controllability and the realized
$n$ per class is reported with every run (§8). Image-grounded affect (EmoSet/FI) is used to
validate the probe (§7), not to condition.

## 5. Evaluation

The **content** track renders neutral content under each intended emotion (content $\perp$
emotion by construction); its floor is `raw` (chance) and `scene` (template only). The **text**
track conditions on AffectBench sentences using the **gold** emotion (the classifier's
prediction is logged separately as classification accuracy); valence/arousal are text-grounded
and continuous, and the floor is `shuffled` — condition on a *wrong* emotion and score against
it, so a high value means the modifier overrides the sentence rather than the sentence leaking
through. For each tier we report:

- **Recovery accuracy / macro-F1** — recovered vs. intended emotion, with **bootstrap 95% CIs**.
- **Valence/arousal correlation and error** — Pearson $r$, Spearman $\rho$ (both with bootstrap
  CIs), and **MAE** between intended and probed affect, the last for an interpretable scale. On
  the content track the intended valence/arousal is the per-emotion prior, so this is a
  *between-emotion* check (does conditioning on "joy" raise recovered valence above "sadness"?)
  and the affect-vs-emotion delta only measures palette/lighting cues; the **text track** is
  where continuous, text-grounded valence/arousal makes affect grounding an independent,
  testable variable.
- **CLIP-T** — image/text alignment, to confirm content is preserved.
- **Shuffled-label control** — a one-sided permutation test ($n=2000$) of each tier's recovery
  against randomly reassigned target emotions, reporting the null mean, 95% interval, and p-value.
  This is the circularity baseline: recovery counts only when it clears the shuffled-label null.

Tier differences (**naive** vs **raw**, **emotion** vs **naive**, **emotion** vs **raw**,
**affect** vs **emotion**, **affect** vs **raw**) are tested with a **paired bootstrap** on
per-item recovery, reporting the mean difference, its CI, and a p-value. Each item is rendered
over $S$ seeds (`--seeds`, default $S=3$ in `make reproduce`; the committed pilot uses $S=1$),
and the **same seed is shared across tiers per item** so contrasts are paired on generation noise
and significance is taken over the (item, seed) population, not a single draw.

**Baselines.** The conditioning tiers are an internal ablation, with **`naive`** — appending the
bare emotion word to the content — serving as the minimal "emotion adjective" baseline that any
method must beat. We do not yet compare against external emotion-conditioned generators
(EmoGen [@yang2024emogen], EmotiCrafter [@yang2024emoticrafter], CoEmoGen [@li2025coemogen]); the
harness is built for it (`--diffusion-model`, `--style`, `--probe` swap the system under test
while holding the protocol fixed), but the shipped runs cover one generator, so the ablation
shows internal monotonicity, not yet a ranked contribution against the field. Cross-system
comparison under this protocol is the natural next artifact.

Generator: SD-Turbo. Probe: CLIP ViT-B/32. The full run manifest — library
versions, git SHA, device, dtype, model revisions, and the benchmark hash — is logged to
`results/paper/results.json`.

## 6. Results

Tables 1 and 2 are generated by `scripts/report.py` from the committed run (`results/paper/`) and
injected below; per-tier confusion matrices and valence/arousal scatters are written to
`results/paper/figures/`. The committed numbers are the CPU pilot (`make pilot`); the powered
configuration is `make reproduce` (see below). `make paper` regenerates the tables from whichever
run is in `results/paper/`.

<!--TABLES-->
<!-- generated by scripts/report.py; do not edit by hand -->
**Table 1.** Affect recovery by conditioning condition.

| Condition | Accuracy [95% CI] | Macro-F1 | Valence ρ†‡ | Arousal ρ†‡ | CLIP-T | n |
|---|---|---|---|---|---|---|
| raw | 0.143 [0.000, 0.357] | 0.038 | 0.076 [-0.47, 0.55] | 0.546 [-0.00, 0.90] | 0.280 | 14 |
| emotion | 0.214 [0.000, 0.429] | 0.112 | 0.474 [-0.03, 0.78] | 0.474 [-0.03, 0.80] | 0.274 | 14 |
| affect | 0.214 [0.000, 0.429] | 0.133 | 0.241 [-0.30, 0.65] | 0.618 [0.19, 0.86] | 0.270 | 14 |
| scene | 0.286 [0.000, 0.575] | 0.184 | 0.414 [-0.63, 1.00] | 0.582 [-0.43, 0.99] | – | 7 |

Chance = 0.143 (1/7); majority-class baseline = 0.143. A probe collapsed onto one label scores here, so recovery is only informative *above* it.
**Probe health:** the probe used 2/7 emotion labels across the conditioning tiers, predicting 'neutral' for 90% of items — every recovery number must be read against this degeneracy.
**Shuffled-label control:** one-sided permutation test of recovery vs randomly reassigned target emotions (null mean 0.142) — emotion p=0.23, affect p=0.14. A p near 1 means recovery is indistinguishable from the circularity baseline, i.e. not above chance label agreement.
† On neutral content the intended valence/arousal is the per-emotion prior, so these correlations reflect between-emotion separation, not within-emotion grounding.
‡ ρ is shown with its bootstrap 95% CI; at small n the interval spans most of [-1, 1], so the point estimate should not be over-read.

**Table 2.** Paired contrasts on per-item recovery (bootstrap).

| Contrast | Δ accuracy | 95% CI | p |
|---|---|---|---|
| emotion vs raw | +0.071 | [0.000, 0.214] | 0.255 |
| affect vs emotion | +0.000 | [0.000, 0.000] | 1.000 |
| affect vs raw | +0.071 | [0.000, 0.214] | 0.255 |

<!--/TABLES-->

The committed run is a small CPU **pilot** (256-px, 2 content subjects, single seed →
$n=14$ per tier), produced by `make pilot` at the recorded git SHA; full provenance is in the
manifest. It is reported not as evidence *for* controllability but as a demonstration that the
protocol, its floors, and its diagnostics run end-to-end — and, just as importantly, that the
**instrument is too weak to draw any positive conclusion from**. Three things to read together:

1. **The probe has collapsed in-domain.** The `probe_health` diagnostic (reported with every run)
   shows the CLIP probe uses only a handful of the seven labels across the conditioning tiers,
   predicting `neutral` for the large majority of generated scenes. This single fact governs
   everything below.
2. **The floors are therefore necessary but not sufficient.** `raw` sits at chance ($0.143$) and
   the `scene` positive control is the highest condition ($0.286$) — consistent with the design,
   but note that a probe collapsed onto one label *also* scores chance on `raw` by construction.
   We report the **majority-class baseline** beside the accuracy precisely so this cannot be
   mistaken for the floors discriminating on their own. `scene` $>$ `raw` is the one piece of
   evidence the probe is not *entirely* blind.
3. **The "lift" is a single item and not significant.** `emotion` over `raw` is $+0.071$
   ($p=0.255$): one extra correct image out of fourteen. `affect` adds nothing over `emotion`
   ($\Delta=0$) — expected, since on neutral content the affect cues are a deterministic function
   of the emotion prior (the affect grounding is an independent variable only on the text track).
4. **No tier beats the shuffled-label control.** The permutation test against randomly reassigned
   targets gives `emotion` $p=0.23$, `affect` $p=0.14$, `scene` $p=0.14$ (null mean $0.142$):
   recovery is statistically indistinguishable from the circularity baseline. So the pilot's
   numbers are not even evidence of non-circular signal, let alone of controllability.

No claim about controllability survives this: a weak-or-null effect is exactly what a collapsed
probe on an underpowered pilot must produce. The honest order of operations is to fix the probe
(§7) *before* spending the powered configuration (`make reproduce`: 512-px, 20 subjects, 3 seeds
→ $n=420$), which is wired but deliberately unrun until the instrument can see more than two of
seven emotions in the domain it is used.

## 7. Probe validation and human study

The probe is a proxy for perceived emotion, so we measure how far it can be trusted — and we are
careful to distinguish *where* it is measured, because a ceiling estimated out of domain does not
describe the operating error.

**Out-of-domain (faces).** `novavision.eval.validate_probe` runs the default CLIP ViT-B/32 probe
on a held-out labelled emotion set ($n=200$, facial-expression imagery) and reports its accuracy
and confusion. The probe recovers the Ekman emotion at only **29.0% accuracy (macro-F1 0.22)** —
barely twice chance. Per-class recall is bimodal: usable for `neutral` (0.81) and `anger` (0.61)
but **near-random for surprise (0.04), fear (0.06), sadness (0.08), and disgust (0.13)**. But this
set is *faces*, not the diffusion-generated *scenes* the benchmark actually probes, so 29% is an
optimistic out-of-domain proxy, not the operating error.

**In-domain (generated scenes).** The operating error is worse. The `probe_health` diagnostic
emitted with every run shows that, on the scenes the pipeline generates, the CLIP probe
**collapses onto `neutral`** — it uses only a small fraction of the seven labels and assigns
`neutral` to the large majority of images. A classifier that resolves essentially two of seven
categories in the measurement domain cannot, even in principle, certify emotional controllability;
this is why §6 reports a null and why the floors, while necessary, are not sufficient on their own.
`make validate-probe-scene` runs the same labelled-set check in domain on EmoSet to quantify the
ceiling directly; an independent non-CLIP probe (`--probe hf`, `make robustness`) and a stronger
backbone (ViT-L/14) slot into the same interface and are the prerequisite for any powered run.

`novavision.eval.human_study` adds the human leg — regenerate a stratified sample, collect labels
from three or more raters, report human-vs-probe Cohen's $\kappa$ — and is wired but unrun (needs
raters).

## 8. Limitations

- **The instrument is the binding limitation.** CLIP recovery is a proxy for human perception,
  and §7 shows it is weak out of domain (29.0% on faces) and *degenerate in domain* — it collapses
  onto `neutral` on generated scenes. Until a probe is validated to resolve more than two of seven
  emotions in domain, no recovery number — including a null — is interpretable as controllability,
  and the powered run is deliberately withheld.
- **A chance-level result is uninformative on its own.** A probe collapsed onto one label scores
  at the majority-class baseline (= chance on balanced labels) regardless of the image, so we
  report that baseline and the `probe_health` collapse diagnostic beside every number; "raw at
  chance" is consistent with both a working protocol *and* a broken probe.
- Mood modifiers and the CLIP probe share affective vocabulary; the `raw` control and `scene`
  floor bound how much recovery this could explain, and an independent non-CLIP probe
  (`--probe hf`) removes the residual overlap once validated against a labelled image set.
- On the content track the intended valence/arousal is the per-emotion prior, so its VA
  correlation is between-emotion and the `affect`-vs-`emotion` contrast tests palette/lighting
  only; within-emotion, text-grounded VA is an independent variable on the text track. VA
  correlations are reported with bootstrap CIs, which at pilot $n$ span most of $[-1, 1]$.
- This is a protocol and harness with a single-generator, single-probe pilot, not a
  system-ranking benchmark; the only baseline is the internal `naive` (emotion-adjective) tier,
  and cross-system comparison against EmoGen/EmotiCrafter/CoEmoGen is supported but not yet run.
- **Dataset bias.** The text track derives from GoEmotions, which is Reddit-English with modest
  Ekman-level inter-annotator agreement ($\kappa \approx 0.33$–$0.44$) and demographic skew;
  AffectBench inherits this. The realized per-class $n$ is reported in the build manifest and is
  small for the scarce classes, so text-track numbers should be read as in-domain and underpowered
  for rare emotions.
- **Statistical power.** The committed pilot uses a single seed ($S=1$); the harness loops $S$
  seeds and pairs contrasts on (item, seed), but variance across seeds is only estimated once the
  powered configuration ($S=3$) is run.
- The Ekman set omits mixed and compound affect, and `disgust` is the scarcest class under the
  Ekman collapse of GoEmotions.

## 9. Reproducibility

`uv pip install -r requirements.lock` pins the exact environment; every model is pinned to a
commit revision (`novavision/config.py`) and the dataset to a revision. Runs are globally seeded
(`novavision.determinism`); cross-device bit-exactness is not guaranteed, so each run declares its
device and dtype in the manifest. The committed `results/paper/` artifact is produced by
`make pilot` (the CPU pilot in §6) and `make reproduce` is the powered configuration for a GPU
machine; `make paper` regenerates the tables and figures from whichever is present, and
`make validate-probe-scene` records the in-domain probe ceiling. The deterministic core and the
real import/eval path are both exercised in CI.

## References

See `references.bib`.
