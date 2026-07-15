"""
Tests for the SnapshotRegistry and DatasetSnapshot (WP3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def registry(tmp_path):
    from faulttrace_data.snapshot import SnapshotRegistry
    return SnapshotRegistry(tmp_path / "snapshots.jsonl")


@pytest.fixture
def sample_snapshot():
    from faulttrace_data.snapshot import DatasetSnapshot, MissingnessSummary, _fingerprint_path
    return DatasetSnapshot(
        snapshot_id="snap_test_001",
        dataset_id="test_dataset",
        source_type="amazon_jsonl",
        source_path_fingerprint="a" * 64,
        row_count=100,
        accepted_count=95,
        rejected_count=3,
        duplicate_count=2,
        malformed_count=0,
        canonical_content_hash="b" * 64,
        ingestion_config_hash="c" * 64,
        raw_content_hash="d" * 64,
    )


class TestSnapshotRegistry:
    def test_register_and_list(self, registry, sample_snapshot):
        registry.register(sample_snapshot)
        snapshots = registry.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].snapshot_id == "snap_test_001"

    def test_list_filter_by_dataset(self, registry, sample_snapshot):
        registry.register(sample_snapshot)

        from faulttrace_data.snapshot import DatasetSnapshot
        other = DatasetSnapshot(
            snapshot_id="snap_other_001",
            dataset_id="other_dataset",
            source_type="amazon_jsonl",
            source_path_fingerprint="e" * 64,
            row_count=50,
            accepted_count=50,
        )
        registry.register(other)

        test_snaps = registry.list_snapshots(dataset_id="test_dataset")
        assert len(test_snaps) == 1
        assert test_snaps[0].dataset_id == "test_dataset"

    def test_inspect_found(self, registry, sample_snapshot):
        registry.register(sample_snapshot)
        found = registry.inspect("snap_test_001")
        assert found is not None
        assert found.row_count == 100

    def test_inspect_not_found(self, registry):
        result = registry.inspect("nonexistent")
        assert result is None

    def test_deactivate(self, registry, sample_snapshot):
        registry.register(sample_snapshot)
        success = registry.deactivate("snap_test_001")
        assert success is True

        # After deactivation, active_only list should be empty
        active = registry.list_snapshots(active_only=True)
        assert len(active) == 0

        # But inactive list should still have it
        all_snaps = registry.list_snapshots(active_only=False)
        assert len(all_snaps) == 1
        assert all_snaps[0].active is False

    def test_deactivate_nonexistent(self, registry):
        success = registry.deactivate("does_not_exist")
        assert success is False

    def test_validate_no_parquet(self, registry, sample_snapshot, tmp_path):
        """Validate should fail gracefully if parquet_root doesn't exist."""
        registry.register(sample_snapshot)
        result = registry.validate("snap_test_001", data_root=tmp_path)
        # No parquet_root is set → no parquet check → skipped
        assert result["status"] in ("valid", "invalid")
        assert result["snapshot_id"] == "snap_test_001"

    def test_registry_file_is_jsonl(self, registry, sample_snapshot):
        registry.register(sample_snapshot)
        lines = [l for l in registry.registry_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["snapshot_id"] == "snap_test_001"

    def test_corrupt_line_skipped(self, registry, sample_snapshot):
        """Corrupt lines in JSONL should be silently skipped."""
        registry.register(sample_snapshot)
        # Append a corrupt line
        with open(registry.registry_path, "a") as f:
            f.write("not valid json\n")
        # Should still return the valid snapshot
        snapshots = registry.list_snapshots()
        assert len(snapshots) == 1


class TestDatasetSnapshot:
    def test_snapshot_hash_stable(self, sample_snapshot):
        h1 = sample_snapshot.snapshot_hash()
        h2 = sample_snapshot.snapshot_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_make_snapshot_id_deterministic(self):
        from faulttrace_data.snapshot import DatasetSnapshot
        sid1 = DatasetSnapshot.make_snapshot_id("ds1", "a" * 64, "b" * 64)
        sid2 = DatasetSnapshot.make_snapshot_id("ds1", "a" * 64, "b" * 64)
        assert sid1 == sid2
        assert len(sid1) == 16

    def test_snapshot_id_differs_by_dataset(self):
        from faulttrace_data.snapshot import DatasetSnapshot
        sid1 = DatasetSnapshot.make_snapshot_id("ds1", "a" * 64, "b" * 64)
        sid2 = DatasetSnapshot.make_snapshot_id("ds2", "a" * 64, "b" * 64)
        assert sid1 != sid2

    def test_schema_version_present(self, sample_snapshot):
        from faulttrace_data.snapshot import SNAPSHOT_SCHEMA_VERSION
        assert sample_snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION

    def test_missingness_summary_defaults(self, sample_snapshot):
        assert sample_snapshot.missingness.price_null_count == 0
        assert sample_snapshot.missingness.price_null_ratio == 0.0

    def test_model_dump_json_roundtrip(self, sample_snapshot):
        dumped = sample_snapshot.model_dump_json()
        from faulttrace_data.snapshot import DatasetSnapshot
        reloaded = DatasetSnapshot.model_validate_json(dumped)
        assert reloaded.snapshot_id == sample_snapshot.snapshot_id
        assert reloaded.row_count == sample_snapshot.row_count


class TestPathFingerprinting:
    def test_fingerprint_no_absolute_path(self, tmp_path):
        from faulttrace_data.snapshot import _fingerprint_path
        data_root = tmp_path / "data"
        data_root.mkdir()
        source = data_root / "raw" / "reviews.jsonl"
        fp = _fingerprint_path(source, data_root)
        # Fingerprint is SHA-256 of relative path — not the absolute path
        assert len(fp) == 64
        assert str(source) not in fp  # Hash, not path string

    def test_fingerprint_same_relative_same_hash(self, tmp_path):
        from faulttrace_data.snapshot import _fingerprint_path
        data_root = tmp_path / "data"
        data_root.mkdir()
        source1 = data_root / "raw" / "reviews.jsonl"
        source2 = data_root / "raw" / "reviews.jsonl"
        assert _fingerprint_path(source1, data_root) == _fingerprint_path(source2, data_root)

    def test_fingerprint_different_paths_different_hash(self, tmp_path):
        from faulttrace_data.snapshot import _fingerprint_path
        data_root = tmp_path / "data"
        data_root.mkdir()
        source1 = data_root / "reviews_v1.jsonl"
        source2 = data_root / "reviews_v2.jsonl"
        assert _fingerprint_path(source1, data_root) != _fingerprint_path(source2, data_root)
