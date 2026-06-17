# Security

## Reporting

Please report vulnerabilities privately via a GitHub security advisory on this repository.
Do not open public issues for security problems.

## Secrets & configuration

- No secrets are committed. Configuration is read from environment variables; see `.env.example`.
- `HF_TOKEN` is needed only for the hosted `hf-api` backend.
- The full git history is scanned for committed secrets on every push (`gitleaks` in CI).

## Hardening defaults

- **Localhost by default.** Both entry points (`server.py`, `app.py`) bind `127.0.0.1`. A public
  bind (`0.0.0.0`) requires an explicit `NOVA_PUBLIC=1` opt-in (auto-enabled only inside a
  Hugging Face Spaces sandbox). The shared rule lives in `novavision/serving.py` so the two
  entry points cannot disagree.
- **Generate route is protected.** `/api/generate` enforces a per-IP rate limit
  (`NOVA_RATE_LIMIT`, default 30/min), a concurrency cap (`NOVA_MAX_CONCURRENCY`, default 2, which
  sheds load rather than queueing), and an optional bearer token (`NOVA_API_TOKEN`) — set the
  token whenever you bind publicly.
- **No repo-root exposure.** Flask serves only the dedicated `static/` directory, never the
  project root, so source, configs, and data are not downloadable.
- **CORS** is disabled by default (same-origin). Set `CORS_ORIGINS` to allow specific domains.
- Request bodies are capped at 64 KB and input text at 2000 characters.
- **No unsafe deserialization.** Model weights load only via Hugging Face `from_pretrained`
  (safetensors); `pickle.load`/`torch.load`/`weights_only=False`/`shell=True` are blocked by a
  CI test (`tests/test_security.py`).
- API errors return generic messages; details are logged server-side only.
- **CI** runs with least-privilege `permissions: contents: read`; no workflow has write scope.

## Sample data

`tests/fixtures/affectbench_sample.csv` is fictional, hand-authored example content — see
`data/README.md`. No personal or production data is included anywhere in this repository.
