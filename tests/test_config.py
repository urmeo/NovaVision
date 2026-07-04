from novavision.config import Settings


def test_settings_honor_nova_prefix(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "hf-api")
    assert Settings().backend == "hf-api"


def test_bare_backend_is_ignored(monkeypatch):
    # A generic BACKEND in a shell or base image must not override the run.
    monkeypatch.delenv("NOVA_BACKEND", raising=False)
    monkeypatch.setenv("BACKEND", "diffusers")
    assert Settings().backend == "null"  # default, bare name ignored


def test_unknown_env_vars_ignored(monkeypatch):
    monkeypatch.setenv("NOVA_NOT_A_FIELD", "x")
    Settings()  # extra="ignore": must not raise
