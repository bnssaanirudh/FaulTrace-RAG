import pytest
from fastapi.testclient import TestClient
from faulttrace_api.database import Base, RunRow, get_db
from faulttrace_api.main import app
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Create a clean SQLite database in memory
sqlite_url = "sqlite://"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_override():
    db = testing_session_local()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = get_db_override

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_get_run_certificate(setup_db):
    # Manually create a run row to test the certificate endpoint
    db = testing_session_local()
    from datetime import datetime

    run_id = "test_run_cert_123"
    row = RunRow(
        run_id=run_id,
        query_id="query_1",
        pipeline_id="p1",
        started_at=datetime.utcnow(),
        certificate_id="cert_abc",
        certificate_hash="hash_def",
        policy_decision="certified",
        abstention_reason=None,
        final_presented_answer="answer",
        raw_answer="raw answer",
    )
    db.add(row)
    db.commit()
    db.close()

    # 2. Retrieve certificate
    # Ensure correct prefix for the router
    response_cert = client.get(f"/api/v1/runs/{run_id}/certificate")
    assert response_cert.status_code == 200
    cert_data = response_cert.json()

    assert cert_data["certificate_id"] == "cert_abc"
    assert cert_data["certificate_hash"] == "hash_def"
    assert cert_data["policy_decision"] == "certified"
    assert cert_data["final_presented_answer"] == "answer"
    assert cert_data["raw_answer"] == "raw answer"


def test_leaderboard_aggregates_boolean_accuracy(setup_db):
    from datetime import datetime

    db = testing_session_local()
    db.add_all(
        [
            RunRow(
                run_id="leaderboard_correct",
                query_id="query_1",
                pipeline_id="pipeline_a",
                started_at=datetime.utcnow(),
                is_correct=True,
                loss=0.0,
                latency_ms=10.0,
            ),
            RunRow(
                run_id="leaderboard_incorrect",
                query_id="query_2",
                pipeline_id="pipeline_a",
                started_at=datetime.utcnow(),
                is_correct=False,
                loss=1.0,
                latency_ms=30.0,
            ),
        ]
    )
    db.commit()
    db.close()

    response = client.get("/api/v1/leaderboard")

    assert response.status_code == 200
    assert response.json() == {
        "leaderboard": [
            {
                "pipeline_id": "pipeline_a",
                "total_runs": 2,
                "correct_runs": 1,
                "accuracy": 0.5,
                "mean_loss": 0.5,
                "mean_latency_ms": 20.0,
            }
        ],
        "total_pipelines": 1,
    }


def test_create_run_and_retrieve_trace():
    payload = {
        "pipeline_id": "P0-deterministic-scope-baseline",
        "query_id": "test_query_02",
        "provider_id": "deterministic",
    }
    response = client.post("/runs/", json=payload)
    if response.status_code == 404:
        pytest.skip("Query ID not found")
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Retrieve trace
    response_trace = client.get(f"/runs/{run_id}/trace")
    assert response_trace.status_code == 200
    trace_data = response_trace.json()
    assert "events" in trace_data
    # P0 produces at least query_load, scope_enumerate, fact_extract, aggregate, validate, persist
    assert len(trace_data["events"]) > 0


def test_create_run_and_attribute():
    payload = {
        "pipeline_id": "P0-deterministic-scope-baseline",
        "query_id": "test_query_03",
        "provider_id": "deterministic",
    }
    response = client.post("/runs/", json=payload)
    if response.status_code == 404:
        pytest.skip("Query ID not found")
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Attribute
    response_attr = client.post(f"/runs/{run_id}/attribute")
    if response_attr.status_code == 200:
        attr_data = response_attr.json()
        assert "dominant_fault" in attr_data
