from fastapi.testclient import TestClient

from clinagent.api.app import create_app
from clinagent.settings import Settings


def _client(qa, key=""):
    return TestClient(create_app(Settings(api_key=key), qa=qa))


def test_ask_returns_cited_answer(qa):
    r = _client(qa).post(
        "/v1/ask", json={"question": "What oxygen saturation target in acute asthma?"}
    )
    body = r.json()
    assert r.status_code == 200 and body["grounded"]
    assert body["citations"][0]["text"]
    assert body["disclaimer"] and body["trace_id"]
    assert {s["name"] for s in body["spans"]} >= {"retrieve", "verify"}


def test_patient_question(qa):
    r = _client(qa).post(
        "/v1/ask",
        json={"question": "Cefazolin dose for this patient's weight?", "patient_id": "pt-004"},
    )
    assert "Observation/obs-030" in [c["ref"] for c in r.json()["citations"]]


def test_api_key_enforced(qa):
    client = _client(qa, key="s3cret")
    assert client.post("/v1/ask", json={"question": "asthma oxygen target"}).status_code == 401
    ok = client.post(
        "/v1/ask",
        json={"question": "asthma oxygen target"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 200


def test_invalid_patient_id_rejected(qa):
    r = _client(qa).post("/v1/ask", json={"question": "dose?", "patient_id": "../../etc/passwd"})
    assert r.status_code == 422


def test_health_ready_metrics(qa):
    c = _client(qa)
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").json()["chunks"] > 0
    c.post("/v1/ask", json={"question": "capital of France?"})
    assert "clinagent_requests_total" in c.get("/metrics").text
