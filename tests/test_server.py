import pytest

pytest.importorskip("flask")

from PIL import Image  # noqa: E402

import server  # noqa: E402
from novavision.affect.analyzer import EmotionAnalysis  # noqa: E402
from novavision.pipeline import Result  # noqa: E402


class FakeAnalyzer:
    def analyze(self, text):
        return EmotionAnalysis("joy", 0.91, 0.8, 0.7, 1.0, {"joy": 0.91, "sadness": 0.09})


class FakePipeline:
    analyzer = FakeAnalyzer()

    def auto_run(self, text, style="artistic", seed=0):
        image = Image.new("RGB", (8, 8), (10, 20, 30))
        return Result(image, "prompt", self.analyzer.analyze(text), "affect", style, seed, "null")


@pytest.fixture
def client():
    server._pipeline = FakePipeline()
    return server.app.test_client()


def test_analyze_ok(client):
    resp = client.post("/api/analyze", json={"text": "i feel great"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["primary_emotion"] == "joy"
    assert data["emotions"][0]["name"] == "joy"


def test_analyze_short_text(client):
    assert client.post("/api/analyze", json={"text": "hi"}).status_code == 400


def test_generate_ok(client):
    resp = client.post(
        "/api/generate", json={"text": "i feel great", "style": "artistic", "seed": 5}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["image"].startswith("data:image/png;base64,")
    assert data["seed"] == 5


def test_index_served(client):
    assert client.get("/").status_code == 200
