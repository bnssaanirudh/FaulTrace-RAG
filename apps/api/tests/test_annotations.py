from fastapi.testclient import TestClient
from faulttrace_api.main import app

client = TestClient(app)


def test_annotation_routes():
    # 1. Create a task
    response = client.post(
        "/api/v1/annotations/tasks?world_id=w1&query_id=q1", json={"question": "Test?"}
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # 2. List tasks
    response = client.get("/api/v1/annotations/tasks")
    assert response.status_code == 200
    assert len(response.json()) > 0

    # 3. Assign task
    response = client.post(f"/api/v1/annotations/assignments?task_id={task_id}&user_id=test_user")
    assert response.status_code == 200
    assignment_id = response.json()["assignment_id"]

    # 4. Submit
    response = client.post(
        f"/api/v1/annotations/assignments/{assignment_id}/submit?time_spent=10",
        json={"text": "Looks good"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["text"] == "Looks good"
