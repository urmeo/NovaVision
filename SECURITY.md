# Security

## Reporting

Please report vulnerabilities privately via a GitHub security advisory on this repository.
Do not open public issues for security problems.

## Secrets & configuration

- No secrets are committed. Configuration is read from environment variables.
- `HF_TOKEN` is needed only for the hosted `hf-api` backend.

## Hardening defaults

- **CORS** is disabled by default (same-origin). Set `CORS_ORIGINS` to allow specific domains.
- The dev server binds to `127.0.0.1` by default (`HOST`/`PORT` configurable).
- Request bodies are capped at 64 KB and input text at 2000 characters.
- API errors return generic messages; details are logged server-side only.

## Sample data

`tests/fixtures/affectbench_sample.csv` is fictional, hand-authored example content — see
`data/README.md`. No personal or production data is included anywhere in this repository.
