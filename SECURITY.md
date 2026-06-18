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

- **Localhost by default.** Both entry points (`server.py`, `app.py`) bind `127.0.0.1`. A public
  bind (`0.0.0.0`) requires an explicit `NOVA_PUBLIC=1` opt-in (auto-enabled only inside a
  Hugging Face Spaces sandbox). The shared rule lives in `novavision/serving.py` so the two
  entry points cannot disagree.
- **Generate route is protected.** `/api/generate` enforces a per-IP rate limit
  (`NOVA_RATE_LIMIT`, default 30/min), a concurrency cap (`NOVA_MAX_CONCURRENCY`, default 2, which
  sheds load rather than queueing), and an optional bearer token (`NOVA_API_TOKEN`) — set the
  token whenever you bind publicly. The rate-limit key is `request.remote_addr`;
  `X-Forwarded-For` is trusted only when `NOVA_TRUST_PROXY=1` (set it only behind a real reverse
  proxy, since the header is otherwise client-spoofable).
- **No repo-root exposure.** Flask serves only the dedicated `static/` directory, never the
  project root, so source, configs, and data are not downloadable.
- **CORS** is disabled by default (same-origin). Set `CORS_ORIGINS` to allow specific domains.
- Request bodies are capped at 64 KB and input text at 2000 characters.
- **No unsafe deserialization.** Model weights load only via Hugging Face `from_pretrained`
  (safetensors); `pickle.load`/`torch.load`/`weights_only=False`/`shell=True` are blocked by a
  CI test (`tests/test_security.py`).
- API errors return generic messages; details are logged server-side only.

## Secrets and CI

- No secrets are committed. Configuration is read from environment variables; see `.env.example`.
- The full git history is scanned for committed secrets on every push (`gitleaks` in CI), and the
  dependency chain is audited by `pip-audit`.
- CI runs with least-privilege `permissions: contents: read`; no workflow has write scope.
