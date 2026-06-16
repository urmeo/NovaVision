# A Reproducible Protocol for Measuring Emotion Controllability in Text-to-Image Generation

**Urme Bose** (`urme-b`)

> Working draft. The Results tables and figures are produced by the reproducible
> pipeline (`make reproduce && make paper`), which generates `paper/tables.md` and
> fills the Results block below — never hand-written. Until a canonical run is
> committed, that block shows the command that populates it.

## Abstract

Emotional text-to-image generation is usually judged by inspection, which conflates the emotion
*intended* by the prompt with the emotion an image actually conveys. A natural automatic
alternative — condition on an emotion, then recover it from the image with a zero-shot probe —
is easy to get wrong: if the conditioning paints a fixed per-emotion scene and the probe is
written to match it, "recovery" becomes a tautology. We make the protocol honest. Image
*content* is held independent of the intended emotion, emotion enters only as a modifier, and
the headline number is bounded by two floors a tautology cannot beat: a no-emotion control
(chance) and a template-only ceiling (pure scene recognition). We pair this with **AffectBench**,
a frozen, datasheeted benchmark derived from GoEmotions, and report every tier with bootstrap
confidence intervals and a paired significance test. The contribution is the protocol and
artifact — designed around its failure modes — not a single controllability score; we also ship
probe-validation and human-study harnesses so the proxy can be checked against people.

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
as EmoGen [@yang2024emogen] motivates the task. Evaluation uses CLIP zero-shot transfer
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
  prior by coverage $c$).

**Floors.** Two conditions bound the claim: **raw** is the no-emotion control (recovery should
sit at chance, $1/7$); **scene** renders a fixed per-emotion template with *no* content and
measures how much recovery is pure template recognition. A method is only credible if it clears
the control and adds signal over the template.

**Recovery probe.** A swappable `Probe` reads emotion and graded valence/arousal back from the
image. The default is CLIP zero-shot with two fixes over the naive version: logits are scaled by
the model's learned temperature before softmax (so valence/arousal are not squeezed toward zero),
and valence/arousal are read as the expected value over an *ordered ladder* of anchor prompts
rather than a single positive/negative contrast. Probe templates are frozen before measurement.

## 4. AffectBench

The primary track conditions on a bank of twenty affect-neutral content subjects
(`data/content_bank.txt`), so content is independent of emotion by construction. **AffectBench**
backs a secondary *text* track: it is built on demand from GoEmotions [@demszky2020] under the
official Ekman grouping — single-label examples mapped to seven Ekman categories, deduplicated,
sampled per class from the **test** split (so items do not overlap model training splits), and
interleaved so any prefix stays balanced. The build records realized per-class counts and a
content hash; a datasheet (`data/DATASHEET.md`) documents composition, splits, and licensing. A
hand-authored sample ships only as a test fixture and can never produce a reported number —
`load_benchmark` has no default path. Because GoEmotions is *text* emotion, the text track is
scoped to text-prompt controllability; image-grounded affect (EmoSet/FI) is used to validate the
probe (§7).

## 5. Evaluation

The primary track renders neutral content under each intended emotion (content $\perp$ emotion by
construction). For each tier we report:

- **Recovery accuracy / macro-F1** — recovered vs. intended emotion, with **bootstrap 95% CIs**.
- **Valence/arousal correlation** — Pearson $r$ and Spearman $\rho$ between intended and probed
  affect. On neutral content the intended valence/arousal is the per-emotion prior, so this is a
  *between-emotion* check (does conditioning on "joy" raise recovered valence above "sadness"?),
  not within-emotion grounding; we report it but do not headline it. Continuous, text-grounded
  valence/arousal is exercised only on the secondary text track. Because the affect tier's cues
  are derived from that same prior on neutral content, the affect-vs-emotion delta measures the
  marginal value of palette/lighting cues, not of independent affect grounding.
- **CLIP-T** — image/text alignment, to confirm content is preserved.

Tier differences (**emotion** vs **raw**, **affect** vs **emotion**, **affect** vs **raw**) are
tested with a **paired bootstrap** on per-item recovery, reporting the mean difference, its CI,
and a p-value. Each item is rendered over several seeds, shared across tiers for paired
comparison. Generator: SD-Turbo. Probe: CLIP ViT-B/32. The full run manifest — library
versions, git SHA, device, dtype, model revisions, and the benchmark hash — is logged to
`results/paper/results.json`.

## 6. Results

Tables 1 and 2 are generated by `scripts/report.py` from the canonical run (`results/paper/`)
and injected below; per-tier confusion matrices and valence/arousal scatters are written to
`results/paper/figures/`. Run `make reproduce && make paper` to populate them.

<!--TABLES-->
*Run `make reproduce && make paper` to populate Tables 1–2 from the canonical run.*
<!--/TABLES-->

## 7. Probe validation and human study

The probe is a proxy for perceived emotion, and two harnesses are provided to bound how far it
can be trusted. `novavision.eval.validate_probe` runs the probe on a labelled image-emotion set
(FI/EmoSet, mapped to the Ekman labels) and reports accuracy and confusion as the instrument's
known error. `novavision.eval.human_study` regenerates a stratified sample of images
deterministically, collects labels from three or more raters, and reports human-vs-probe
agreement (Cohen's $\kappa$) and human-vs-intended accuracy. Both harnesses ship with tests; the
populated numbers require the labelled set and the rater pass, and are reported alongside the
canonical run. Until then this section states the protocol, not results.

## 8. Limitations

- CLIP recovery is a proxy for human perception; §7 bounds it but a larger human study would
  strengthen it.
- Mood modifiers and probe prompts share affective vocabulary; the shuffled/scene floors and the
  cross-probe check mitigate this, and an independently-trained image-emotion probe would remove
  it entirely.
- On neutral content the intended valence/arousal varies mainly between emotions, so the VA
  correlation should be read within-emotion; we report it but do not headline it.
- The Ekman set omits mixed and compound affect, and `disgust` is the scarcest class under the
  Ekman collapse of GoEmotions.

## 9. Reproducibility

`uv pip install -r requirements.lock` pins the exact environment; every model is pinned to a
commit revision (`novavision/config.py`) and the dataset to a revision. Runs are globally seeded
(`novavision.determinism`); cross-device bit-exactness is not guaranteed, so the canonical run
declares its device and dtype in the manifest. `make reproduce` runs the experiment and
`make paper` regenerates the tables and figures from it. The deterministic core and the real
import/eval path are both exercised in CI.

## References

See `references.bib`.
