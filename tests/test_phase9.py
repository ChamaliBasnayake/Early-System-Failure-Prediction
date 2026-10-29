"""Phase 9 tests: featurisation of raw lines and the API (skipped when the model artefacts are not present)."""
import pytest
from fastapi.testclient import TestClient

from app.backend import settings

ARTEFACTS = (settings.INTERIM / "drain_miner.pkl").exists() and any((settings.CHECKPOINT_DIR / f"{r}.pt").exists() for r in settings.MODEL_RUNS.values())
pytestmark = pytest.mark.skipif(not ARTEFACTS, reason="model artefacts (drain_miner.pkl, checkpoint) not built yet")

GOOD = """- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
- 1117838600 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.43.20.000000 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
- 1117838700 2005.06.03 R63-M1-N8-C:J03-U01 2005-06-03-15.45.00.000000 R63-M1-N8-C:J03-U01 RAS KERNEL INFO generating core.1234
"""


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 't.db'}"
    import importlib; from app.backend import settings as s; importlib.reload(s)
    from app.backend.main import app
    with TestClient(app) as c: yield c


def test_health_and_model(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/model").json(); assert "limitations" in m and m["models"]


def test_predict_normal_invalid_and_empty(client):
    r = client.post("/api/predict", json={"log_text": GOOD, "lead_time": 5, "mc_samples": 3}); assert r.status_code == 200, r.text
    body = r.json(); assert set(body["units_detected"]) == {"R02-M1", "R63-M1"} and len(body["results"]) == 2
    res = {x["unit"]: x for x in body["results"]}; assert res["R02-M1"]["n_local"] == 2 and 0 <= res["R02-M1"]["score"] <= 1 and res["R02-M1"]["mc_spread"] is not None
    assert client.post("/api/predict", json={"log_text": "", "lead_time": 5}).status_code == 400
    assert client.post("/api/predict", json={"log_text": "hello world\nthis is not a log\n", "lead_time": 5}).status_code == 422
    assert client.post("/api/predict", json={"log_text": GOOD, "lead_time": 7}).status_code == 400
    assert client.post("/api/predict", json={"log_text": GOOD, "unit": "R99-M9", "lead_time": 5}).status_code == 422


def test_feedback_and_history(client):
    pid = client.post("/api/predict", json={"log_text": GOOD, "unit": "R02-M1", "lead_time": 5, "mc_samples": 0}).json()["results"][0]["prediction_id"]
    assert client.post("/api/feedback", json={"prediction_id": pid, "correct": False, "note": "test"}).json()["status"] == "stored"
    h = client.get("/api/history").json(); assert h[0]["id"] == pid and h[0]["feedback"]["correct"] is False


def test_explain_returns_attributions(client):
    r = client.post("/api/explain", json={"log_text": GOOD, "unit": "R02-M1", "lead_time": 5}); assert r.status_code == 200, r.text
    ex = r.json(); assert len(ex["items_local"]) >= 1 and len(ex["items_global"]) >= 1 and len(ex["scalars"]) == 17
    assert all("text" in d for d in ex["items_local"]) and abs(ex["completeness_gap"]) < 0.5
    assert any(l["item"] is not None for l in ex["lines"])
    assert client.post("/api/explain", json={"log_text": GOOD, "lead_time": 5}).status_code == 400          # unit required
