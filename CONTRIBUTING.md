# Contributing

Thanks for your interest. NovaVision is a research artifact; contributions that improve
**rigor, reproducibility, or clarity** are especially welcome.

## Setup

```bash
make setup     # core + dev deps
make test      # 100 tests, no models needed
make lint      # ruff check + format
```

## Before opening a PR

- `make test && make lint` must pass; CI runs them on Python 3.9–3.12 with a coverage gate,
  plus `mypy`, a secret scan (`gitleaks`), and a dependency audit (`pip-audit`).
- Keep the deterministic core dependency-light: heavy deps (torch, transformers, diffusers,
  CLIP) are imported **inside** the functions that use them, never at module top.
- Add or update a test for any behavior change. Statistics live in `novavision/eval/metrics.py`
  and are unit-tested to the edge cases (nan, ties, small-n).
- Commit messages are short and human (`updated paper`, `cross-split dedup`); one logical change
  per commit.

## Scope guardrails (the project's integrity rules)

- **Never let a reported number come from a non-reproducible path.** `load_benchmark` has no
  default; the test fixture can never back a result.
- **Don't weaken the controls.** The floors (`raw`, `scene`), the shuffled-label control, and the
  probe-collapse diagnostic exist to keep the headline honest — extend them, don't bypass them.
- **Keep model/data licenses straight** (see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md));
  the repo's MIT covers code only.

## Reporting security issues

See [SECURITY.md](SECURITY.md) — use a private advisory, not a public issue.
