from pathlib import Path

import pytest
from faulttrace_data.benchmarks.provenance import (
    build_benchmark_manifest,
    find_query_split_overlap,
)


def test_manifest_hashes_explicit_files_and_detects_split_leakage(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"id":"q1"}\n', encoding="utf-8")
    test.write_text('{"id":"q2"}\n', encoding="utf-8")

    manifest = build_benchmark_manifest(
        dataset_id="fixture",
        root=tmp_path,
        split_files={"train": [train], "test": [test]},
        source_url="https://example.invalid/dataset",
        license_name="test-only",
        query_ids_by_split={"train": {"q1", "shared"}, "test": {"q2", "shared"}},
    )

    assert len(manifest.files) == 2
    assert len(manifest.snapshot_sha256) == 64
    assert manifest.query_split_overlap == {"shared": ["test", "train"]}


def test_manifest_rejects_implicit_or_outside_files(tmp_path: Path):
    with pytest.raises(ValueError, match="At least one"):
        build_benchmark_manifest(
            dataset_id="empty",
            root=tmp_path,
            split_files={},
            source_url="https://example.invalid",
            license_name="unknown",
        )


def test_overlap_helper_reports_only_cross_split_ids():
    assert find_query_split_overlap({"train": {"a", "b"}, "test": {"b", "c"}}) == {
        "b": ["test", "train"]
    }
