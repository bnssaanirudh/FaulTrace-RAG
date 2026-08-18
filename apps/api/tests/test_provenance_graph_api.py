from pathlib import Path

import pytest
from apps.api.faulttrace_api.main import app
from fastapi.testclient import TestClient
from faulttrace_core.knowledge_graph import ProvenanceGraphBuilder

client = TestClient(app)

@pytest.fixture(scope="module")
def artifacts_test_dir():
    # Setup
    base_dir = Path("artifacts/text_runs")
    base_dir.mkdir(parents=True, exist_ok=True)
    yield base_dir
    # Teardown (could clean up specific files, but let's just leave or delete test files)

def test_provenance_graph_api_retrieves_correct_filename(artifacts_test_dir):
    dataset_id = "testdataset"
    query_id = "testquery123"
    graph_id = f"{dataset_id}_{query_id}"
    filename = f"graph_{graph_id}.json"

    builder = ProvenanceGraphBuilder(graph_id=graph_id, dataset_id=dataset_id)
    builder.add_document_node("n1", "Test Node", dataset_id)
    graph = builder.build()

    graph_path = artifacts_test_dir / filename
    graph.save(graph_path)

    response = client.get(f"/api/v1/evidence/graph/{graph_id}")

    # cleanup
    if graph_path.exists():
        graph_path.unlink()

    assert response.status_code == 200
    data = response.json()
    assert data["graph_id"] == graph_id
    assert data["dataset_id"] == dataset_id
    assert len(data["nodes"]) == 1

def test_provenance_graph_api_fallback_filename(artifacts_test_dir):
    dataset_id = "testdataset"
    query_id = "testqueryfallback"
    graph_id = f"{dataset_id}_{query_id}"
    filename = f"graph_{query_id}.json"  # Old format

    builder = ProvenanceGraphBuilder(graph_id=graph_id, dataset_id=dataset_id)
    builder.add_document_node("n1", "Test Node Fallback", dataset_id)
    graph = builder.build()

    graph_path = artifacts_test_dir / filename
    graph.save(graph_path)

    response = client.get(f"/api/v1/evidence/graph/{graph_id}")

    # cleanup
    if graph_path.exists():
        graph_path.unlink()

    assert response.status_code == 200
    data = response.json()
    assert data["graph_id"] == graph_id

def test_provenance_graph_api_404():
    response = client.get("/api/v1/evidence/graph/nonexistent_123")
    assert response.status_code == 404
    assert "Expected filename: artifacts/text_runs/graph_nonexistent_123.json" in response.json()["message"]
