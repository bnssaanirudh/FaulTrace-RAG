from fastapi.testclient import TestClient
from faulttrace_api.main import app

client = TestClient(app)

def test_security_headers():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Content-Security-Policy") == "default-src 'self'"

def test_path_traversal_blocked():
    # Attempt to ingest a file outside the data root
    payload = {
        "input_path": "../../../etc/passwd",
        "dataset_id": "hack",
        "license_note": "hack"
    }
    response = client.post("/api/v1/datasets/ingest", json=payload)
    assert response.status_code == 400
    assert "Path traversal" in response.json()["message"] or "Invalid path" in response.json()["message"]
