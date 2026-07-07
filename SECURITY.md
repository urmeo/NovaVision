# Security Policy

NovaVision is a research artifact: a localhost-by-default Flask app plus an offline Python
evaluation pipeline. The app binds `127.0.0.1` unless you opt in to a public bind, stores no
user data, and ships no secrets or API keys. It downloads model weights (SD-Turbo, CLIP, the
emotion classifier) from Hugging Face via `from_pretrained`; an `HF_TOKEN` is needed only for
the hosted `hf-api` backend.

## Supported versions

Only the latest main is supported. Fixes land on main; there are no backports.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use the repository's
**Security** tab → **Report a vulnerability** to open a private security advisory.

Expect an acknowledgement within a few days.

## Hardening defaults

- **Localhost by default.** The entry point (`server.py`) binds `127.0.0.1`. A public bind
  (`0.0.0.0`) requires an explicit `NOVA_PUBLIC=1` opt-in (auto-enabled only inside a
  Hugging Face Spaces sandbox). The rule lives in `novavision/serving.py`.
- **Generate route is protected.** `/api/generate` enforces a per-IP rate limit
  (`NOVA_RATE_LIMIT`, default 30/min), a concurrency cap (`NOVA_MAX_CONCURRENCY`, default 2, which
  sheds load rather than queueing), and an optional bearer token (`NOVA_API_TOKEN`): set the
  token whenever you bind publicly. The rate-limit key is `request.remote_addr`;
  `X-Forwarded-For` is trusted only when `NOVA_TRUST_PROXY=1` (set it only behind a real reverse
  proxy, since the header is otherwise client-spoofable).
- **No repo-root exposure.** Flask serves only the dedicated `static/` directory, never the
  project root, so source, configs, and data are not downloadable.
- **CORS** is disabled by default (same-origin). Set `CORS_ORIGINS` to allow specific domains.
- Request bodies are capped at 32 KB (the worst-case wire size of a legal 2000-character
  text, JSON-escaped) and input text at 2000 characters; `style` must be a known preset and
  `seed` a signed 64-bit integer, so no unvalidated input is echoed back.
- **No unsafe deserialization.** Model weights load only via Hugging Face `from_pretrained`
  (safetensors); `pickle.load`/`torch.load`/`weights_only=False`/`shell=True` are blocked by a
  CI test (`tests/test_security.py`).
- API errors return generic messages; details are logged server-side only.
- **Production serving.** `python server.py` runs Flask's built-in development server; for any
  public bind, serve through a real WSGI server instead: `make serve-prod` (gunicorn, installed
  with the `app` extra), with `BIND=0.0.0.0:8000` only behind your own hardening.

## Secrets and CI

- No secrets are committed. Configuration is read from `NOVA_`-prefixed environment variables
  (defined in `novavision/config.py`; serving knobs in `novavision/serving.py`).
- The full git history is scanned for committed secrets on every push (`gitleaks` in CI), and the
  dependency chain is audited by `pip-audit`.
- CI runs with least-privilege `permissions: contents: read`; no workflow has write scope.

## Sample data

`tests/fixtures/affectbench_sample.csv` is fictional, hand-authored example content (see
`data/README.md`). No personal or production data is included anywhere in this repository.
