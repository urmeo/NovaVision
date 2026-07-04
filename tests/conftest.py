import sys
from pathlib import Path

# scripts/ is not a package; make its modules importable by the tests that cover them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
