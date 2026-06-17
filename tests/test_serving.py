from novavision import serving


def test_resolve_host_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("NOVA_PUBLIC", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert serving.resolve_host() == "127.0.0.1"
    assert serving.public_enabled() is False


def test_resolve_host_public_opt_in(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("NOVA_PUBLIC", "1")
    assert serving.resolve_host() == "0.0.0.0"


def test_resolve_host_spaces_sandbox(monkeypatch):
    monkeypatch.delenv("NOVA_PUBLIC", raising=False)
    monkeypatch.setenv("SPACE_ID", "user/novavision")
    assert serving.resolve_host() == "0.0.0.0"


def test_token_ok_disabled_without_env(monkeypatch):
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    assert serving.token_ok(None) is True  # check is opt-in


def test_token_ok_enforced_when_set(monkeypatch):
    monkeypatch.setenv("NOVA_API_TOKEN", "s3cret")
    assert serving.token_ok("s3cret") is True
    assert serving.token_ok("wrong") is False
    assert serving.token_ok(None) is False


def test_rate_limiter_blocks_over_budget():
    rl = serving.RateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=1.0) is True
    assert rl.allow("ip", now=2.0) is False  # third in window -> blocked
    # a different key has its own budget
    assert rl.allow("other", now=2.0) is True


def test_rate_limiter_window_slides():
    rl = serving.RateLimiter(max_requests=1, window_seconds=10)
    assert rl.allow("ip", now=0.0) is True
    assert rl.allow("ip", now=5.0) is False
    assert rl.allow("ip", now=11.0) is True  # earlier hit aged out


def test_concurrency_guard_caps_slots():
    guard = serving.ConcurrencyGuard(max_concurrent=1)
    assert guard.acquire() is True
    assert guard.acquire() is False  # only one slot
    guard.release()
    assert guard.acquire() is True
