from fastapi.testclient import TestClient
from faulttrace_api.main import app

client = TestClient(app)


def test_governance_audit():
    # 1. Create audit log
    response = client.post(
        "/api/v1/governance/audit?user_id=u1&action=test&resource_type=world&resource_id=w1",
        json={"k": "v"},
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == "u1"

    # 2. List audit logs
    response = client.get("/api/v1/governance/audit")
    assert response.status_code == 200
    assert len(response.json()) > 0
