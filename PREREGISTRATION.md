# Preregistration: the powered run

This locks the powered run's hypotheses, sample size, and analysis **before** it is
executed, so the reported result cannot be a post-hoc rationalization. The
committed pilot is a null; this document commits to how the powered run will be
judged, whichever way it comes out. Timestamped by its git commit.

## Hypotheses

- **H1 (conditioning beats chance).** For each conditioning tier (`naive`,
  `emotion`, `affect`), recovery accuracy exceeds the shuffled-label permutation
  null (one-sided, n=2000 permutations).
- **H2 (conditioning beats the template).** `emotion` and `affect` add recovery
  over the `scene` template ceiling on the content track.
- **H0 (the registered null).** Neither holds after multiple-comparison control;
  recovery is indistinguishable from chance label agreement.

## Design

- **Content track:** 20 neutral subjects x 7 Ekman emotions x 3 seeds =
  n=420 per conditioning tier (n=210 for `scene`).
- **Text track:** AffectBench (GoEmotions test split, pinned revision), same seeds.
- **Generator:** as configured (`--diffusion-model`); **probe:** the one that
  clears the in-domain gate (see RUNBOOK step 1), reported with its measured error.
- Seeds are shared across tiers per item, so contrasts are paired on generation
  noise.

## Sample size justification

Fixed **before** running, from `make power` using the measured probe ceiling
(`results/paper/power_analysis.json`): at the ViT-L/14 in-domain ceiling (0.455),
n=420 per tier gives >=95% power to detect even a weak conditioning effect
(strength s=0.2, recovery 0.205) at alpha=0.05. A null at this n is therefore a
properly powered null, not an underpowered one.

## Analysis plan (frozen)

1. Report recovery accuracy per tier with bootstrap 95% CIs and the shuffled-label
   permutation p-value.
2. Apply **Holm-Bonferroni** across the conditioning tiers (family-wise, alpha=0.05).
3. Report **Cohen's h** effect size for every tier-vs-tier contrast.
4. Report the **Rogan-Gladen** probe-error-corrected recovery
   (`scripts/correct_recovery.py`) beside the apparent recovery.
5. Report the `probe_health` collapse diagnostic; a run whose probe collapses
   (uses <= 2 of 7 labels in domain) is reported as instrument failure, not a
   controllability result, regardless of the accuracy numbers.

## Stopping rule

One run at the registered n. No optional stopping, no adding seeds until
significance, no dropping tiers. If the probe collapses in domain, the run is
withheld and the probe is replaced (RUNBOOK step 1), not reinterpreted.

## Deviations

Any departure from this plan is recorded here with its reason before the numbers
are read.
