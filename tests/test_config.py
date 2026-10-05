from novavision.config import Settings


def test_settings_honor_nova_prefix(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "hf-api")
    assert Settings().backend == "hf-api"


def test_bare_backend_is_ignored(monkeypatch):

    monkeypatch.delenv("NOVA_BACKEND", raising=False)
    monkeypatch.setenv("BACKEND", "diffusers")
    assert Settings().backend == "null"


def test_unknown_env_vars_ignored(monkeypatch):
    monkeypatch.setenv("NOVA_NOT_A_FIELD", "x")
    Settings()
