"""Guards that keep verified-clean security properties from regressing."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "novavision", ROOT / "scripts", ROOT / "server.py"]

# Unsafe deserialization: arbitrary pickle/torch.load execute attacker code.
# Model weights must load via HF from_pretrained (safetensors), never these.
_UNSAFE = re.compile(
    r"\bpickle\.loads?\b|\btorch\.load\s*\(|weights_only\s*=\s*False|"
    r"yaml\.load\s*\((?!.*SafeLoader)|\bos\.system\s*\(|shell\s*=\s*True"
)


def _py_files():
    for src in SOURCES:
        if src.is_dir():
            yield from src.rglob("*.py")
        elif src.exists():
            yield src


def test_no_unsafe_deserialization_or_shell():
    offenders = []
    for path in _py_files():
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if _UNSAFE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "unsafe deserialization/shell call introduced:\n" + "\n".join(offenders)


def test_no_default_public_bind():
    # The server must never default to 0.0.0.0; binding is opt-in via novavision.serving.
    server = (ROOT / "server.py").read_text()
    bind = re.compile(r"""getenv\(\s*["']HOST["']\s*,\s*["']0\.0\.0\.0["']""")
    assert not bind.search(server)
    assert "resolve_host()" in server
