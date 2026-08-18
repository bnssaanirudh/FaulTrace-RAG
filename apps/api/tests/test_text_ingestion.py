import json

import pytest
from faulttrace_data.text.pipeline import TextIngestionPipeline


@pytest.fixture
def temp_workspace(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    registry_path = data_root / "manifests" / "text_snapshots.jsonl"
    return data_root, registry_path


def test_jsonl_parsing_and_chunking(temp_workspace):
    data_root, registry_path = temp_workspace

    # Create sample jsonl
    source_dir = data_root / "source"
    source_dir.mkdir()
    sample_file = source_dir / "sample.jsonl"

    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "doc1",
                    "title": "First",
                    "text": "This is a simple text. It has two sentences.",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps({"id": "doc2", "title": "Second", "text": "Another document here. " * 50})
            + "\n"
        )

    pipeline = TextIngestionPipeline(data_root=data_root, registry_path=registry_path)
    snapshot = pipeline.run(
        dataset_id="test_dataset_jsonl",
        source_path=sample_file,
        source_type="jsonl",
        config={
            "chunk_size": 50,
            "overlap": 10,
            "strict_chunk_dedup": False,
            "text_field": "text",
            "id_field": "id",
            "title_field": "title",
        },
    )

    assert snapshot.document_count == 2
    assert snapshot.chunk_count > 2
    assert snapshot.malformed_documents == 0
    assert snapshot.duplicate_documents == 0


def test_csv_and_text_parsing(temp_workspace):
    data_root, registry_path = temp_workspace

    source_dir = data_root / "source_csv"
    source_dir.mkdir()
    sample_file = source_dir / "sample.csv"

    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("id,title,text\n")
        f.write("1,Doc 1,This is the first document text\n")
        f.write("2,Doc 2,This is the second document text\n")

    pipeline = TextIngestionPipeline(data_root=data_root, registry_path=registry_path)
    snapshot = pipeline.run(
        dataset_id="test_dataset_csv",
        source_path=sample_file,
        source_type="csv",
        config={
            "chunk_size": 50,
            "overlap": 10,
            "strict_chunk_dedup": False,
            "text_field": "text",
            "id_field": "id",
            "title_field": "title",
        },
    )

    assert snapshot.document_count == 2


def test_exact_deduplication(temp_workspace):
    data_root, registry_path = temp_workspace

    source_dir = data_root / "source"
    source_dir.mkdir()
    sample_file = source_dir / "dupes.txt"

    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("chunk text. " * 10 + "\n\n" + "chunk text. " * 10)

    pipeline = TextIngestionPipeline(data_root=data_root, registry_path=registry_path)
    snapshot = pipeline.run(
        dataset_id="test_dataset_dupe",
        source_path=sample_file,
        source_type="txt",
        config={"chunk_size": 25, "overlap": 0, "strict_chunk_dedup": True},
    )

    assert snapshot.document_count == 1
    assert snapshot.exact_duplicate_chunks > 0


def test_symlink_escape_protection(temp_workspace):
    data_root, registry_path = temp_workspace

    from fastapi.testclient import TestClient
    from faulttrace_api.main import app

    client = TestClient(app)

    payload = {"input_path": "../../../../etc/passwd", "dataset_id": "hack", "source_type": "txt"}
    response = client.post("/api/v1/datasets/text/ingest", json=payload)
    assert response.status_code == 400
    assert (
        "Path traversal" in response.json().get("message", "")
        or "Path traversal" in response.json().get("detail", "")
        or "Invalid" in response.json().get("message", "")
    )
