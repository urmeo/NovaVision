# Security

## Reporting

Please report vulnerabilities privately via a GitHub security advisory on this repository.
Do not open public issues for security problems.

## Secrets & configuration

- No secrets are committed. Configuration is read from environment variables.
- `HF_TOKEN` is needed only for the hosted `hf-api` backend.

## Hardening defaults

- **CORS** is disabled by default (same-origin). Set `CORS_ORIGINS` to allow specific domains.
- The Flask server binds to `127.0.0.1` by default; the Gradio app binds `0.0.0.0` for Spaces
  but honours `HOST`/`PORT` for a private local run.
- Request bodies are capped at 64 KB and input text at 2000 characters.
- API errors return generic messages; details are logged server-side only.
- **CI** runs with least-privilege `permissions: contents: read`; no workflow has write scope.

## Sample data

`tests/fixtures/affectbench_sample.csv` is fictional, hand-authored example content — see
`data/README.md`. No personal or production data is included anywhere in this repository.
