"""Security helpers shared by the Flask API (`server.py`) and Gradio app (`app.py`).

The two entry points must not disagree on defaults, so the rules live here once:
bind localhost unless an operator explicitly opts into a public bind, and give the
expensive generate route a per-client rate limit, a concurrency cap, and an
optional bearer token. Everything is dependency-free and read from the
environment at call time, so deployment never needs a code change.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

LOCAL_HOST = "127.0.0.1"
PUBLIC_HOST = "0.0.0.0"  # only ever returned behind an explicit opt-in (see resolve_host)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def public_enabled() -> bool:
    """Whether the operator explicitly asked to expose the server publicly.

    True only on an explicit ``NOVA_PUBLIC`` opt-in or inside a genuine Hugging
    Face Spaces sandbox (which sets ``SPACE_ID`` and is meant to be public).
    """
    return _truthy(os.getenv("NOVA_PUBLIC")) or bool(os.getenv("SPACE_ID"))


def resolve_host(default: str = LOCAL_HOST) -> str:
    """Bind localhost by default; bind all interfaces only on explicit opt-in."""
    return PUBLIC_HOST if public_enabled() else default


def token_ok(provided: str | None) -> bool:
    """Constant-time comparison of a presented token against ``NOVA_API_TOKEN``.

    Returns True when no token is configured (the check is opt-in), so local
    development keeps working; once ``NOVA_API_TOKEN`` is set it is enforced.
    """
    expected = os.getenv("NOVA_API_TOKEN", "").strip() or None
    if expected is None:
        return True
    if provided is None:
        return False
    # Compare bytes: hmac.compare_digest rejects non-ASCII str operands.
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        # These knobs guard a public bind; a typo must fail at startup, not
        # silently run with the default.
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


class RateLimiter:
    """Thread-safe per-key sliding-window limiter, no external dependency."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0, gc_threshold: int = 4096):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._gc_threshold = gc_threshold

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Record a hit for ``key`` and report whether it is within the window budget.

        Memory is bounded to clients with a live hit: when the table grows past
        ``gc_threshold`` we sweep keys whose window has fully expired, so a flood of
        distinct (e.g. spoofed) keys cannot grow the limiter without limit.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            cutoff = now - self.window
            if len(self._hits) > self._gc_threshold:
                for k in [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]:
                    del self._hits[k]
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


class ConcurrencyGuard:
    """A non-blocking concurrency cap: a fixed number of slots, no queueing.

    A bounded semaphore would block; here a slot that cannot be acquired
    immediately is refused so an overloaded generator sheds load instead of
    piling up requests behind a slow GPU job.
    """

    def __init__(self, max_concurrent: int):
        self._sem = threading.BoundedSemaphore(max(1, max_concurrent))

    def acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        # A mispaired release is a caller bug; BoundedSemaphore fails loudly on it.
        self._sem.release()
