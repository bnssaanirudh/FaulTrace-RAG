"""Contract tests for every endpoint used by apps/web/lib/api.ts."""

from fastapi.testclient import TestClient
from faulttrace_api.database import Base, get_db
from faulttrace_api.main import app
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.routing import Match

FRONTEND_API_CALLS = [
    ("GET", "/api/v1/health"),
    ("GET", "/api/v1/system/status"),
    ("POST", "/api/v1/demo/seed"),
    ("GET", "/api/v1/worlds"),
    ("GET", "/api/v1/worlds/{world_id}"),
    ("GET", "/api/v1/worlds/{world_id}/records"),
    ("GET", "/api/v1/queries"),
    ("GET", "/api/v1/queries/{query_id}"),
    ("POST", "/api/v1/queries/generate"),
    ("GET", "/api/v1/gold/{query_id}"),
    ("POST", "/api/v1/runs"),
    ("GET", "/api/v1/runs"),
    ("GET", "/api/v1/runs/{run_id}"),
    ("GET", "/api/v1/runs/{run_id}/trace"),
    ("GET", "/api/v1/runs/{run_id}/attribution"),
    ("GET", "/api/v1/runs/{run_id}/certificate"),
    ("GET", "/api/v1/runs/batch-evaluate-policy"),
    ("GET", "/api/v1/policies"),
    ("GET", "/api/v1/policies/{policy_id}"),
    ("GET", "/api/v1/datasets"),
    ("GET", "/api/v1/datasets/{snapshot_id}"),
    ("GET", "/api/v1/datasets/{snapshot_id}/validate"),
    ("GET", "/api/v1/datasets/{snapshot_id}/missingness"),
    ("POST", "/api/v1/datasets/ingest"),
    ("GET", "/api/v1/datasets/text"),
    ("GET", "/api/v1/datasets/text/{snapshot_id}"),
    ("GET", "/api/v1/datasets/text/{snapshot_id}/preview"),
    ("POST", "/api/v1/datasets/text/ingest"),
    ("POST", "/api/v1/experiments/plan"),
    ("POST", "/api/v1/experiments/run"),
    ("GET", "/api/v1/experiments"),
    ("GET", "/api/v1/experiments/{id}"),
    ("POST", "/api/v1/experiments/{id}/cancel"),
    ("POST", "/api/v1/experiments/compare"),
    ("GET", "/api/v1/annotations/tasks"),
    ("POST", "/api/v1/annotations/assignments"),
    ("POST", "/api/v1/annotations/assignments/{assignment_id}/submit"),
]


def _concrete_path(path: str) -> str:
    replacements = {
        "{world_id}": "contract-world",
        "{query_id}": "contract-query",
        "{run_id}": "contract-run",
        "{policy_id}": "strict_exact_v1",
        "{snapshot_id}": "contract-snapshot",
        "{id}": "contract-experiment",
        "{assignment_id}": "contract-assignment",
    }
    for template, value in replacements.items():
        path = path.replace(template, value)
    return path


def test_every_frontend_call_exists_in_openapi():
    schema = app.openapi()
    paths = schema["paths"]
    missing = [
        f"{method} {path}"
        for method, path in FRONTEND_API_CALLS
        if path not in paths or method.lower() not in paths[path]
    ]
    assert missing == []



def test_previously_shadowed_collection_routes_are_reachable():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine)

    def override_db():
        session = local_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        assert client.get("/api/v1/datasets/text").status_code == 200
        assert client.get("/api/v1/runs/batch-evaluate-policy").status_code == 200
    finally:
        app.dependency_overrides.pop(get_db, None)
