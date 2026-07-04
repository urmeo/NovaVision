"""End-to-end golden path: run -> results.json -> paper tables, no models.

Every stage is unit-tested in isolation; this pins the whole chain so a break at
any seam (record schema, summary keys, report renderer) fails one obvious test.
"""

import json

import report  # scripts/ on sys.path via conftest.py

from novavision.eval.probes import Recovery
from novavision.experiments import run
from novavision.taxonomy import EMOTIONS


class _FakeProbe:
    name = "clip:golden"

    def __init__(self, *a, **k):
        pass

    def recover(self, image):
        return Recovery("joy", 0.4, 0.6, {e: 1.0 if e == "joy" else 0.0 for e in EMOTIONS})

    def clip_t(self, image, text):
        return 0.3


def test_run_to_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", _FakeProbe)
    result = run.run_experiment(backend="null", contents=2, seeds=1, out=str(tmp_path))

    payload = json.loads((tmp_path / "results.json").read_text())
    assert set(payload) >= {"manifest", "metrics", "contrasts", "records"}
    assert payload["manifest"]["device_info"]  # provenance present

    # The renderer consumes the committed schema and produces both tables.
    tables = report.render(payload)
    assert "Table 1" in tables and "Table 2" in tables
    for tier in run.CONDITIONS["content"]:
        if tier in result["metrics"]:
            assert tier in tables
