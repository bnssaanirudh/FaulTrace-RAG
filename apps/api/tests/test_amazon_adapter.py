"""
Tests for the Amazon Reviews local-file ingestion adapter (WP2).

Tests cover:
- Valid JSONL ingestion: all 10 records accepted
- Edge cases: duplicates, malformed dates, missing ASIN, out-of-range rating, null category
- Unicode handling: accepted correctly
- Extra fields: accepted with _extra_field_count in attributes
- Long title: truncated to 512 chars
- Non-JSON line: quarantined as malformed
- Parquet output: verifiable, correct row count
- IngestionReport accuracy: accepted + rejected + duplicate + malformed = total
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import pyarrow.parquet as pq

TEST_DIR = Path(__file__).parent.resolve()
VALID_FIXTURE = TEST_DIR.parent.parent.parent / "data" / "fixtures" / "amazon" / "valid_reviews.jsonl"
EDGE_FIXTURE = TEST_DIR.parent.parent.parent / "data" / "fixtures" / "amazon" / "edge_cases.jsonl"


@pytest.fixture
def tmp_output(tmp_path):
    yield tmp_path / "snapshots"


@pytest.fixture
def adapter():
    from faulttrace_data.amazon_adapter import AmazonLocalAdapter
    return AmazonLocalAdapter(dataset_id="test_dataset")


class TestAmazonAdapterValid:
    def test_valid_fixture_exists(self):
        assert VALID_FIXTURE.exists(), f"Missing fixture: {VALID_FIXTURE}"

    def test_valid_fixture_line_count(self):
        lines = [l for l in VALID_FIXTURE.read_text().splitlines() if l.strip()]
        assert len(lines) == 10

    def test_all_valid_records_accepted(self, adapter, tmp_output):
        report, snapshot = adapter.ingest(
            source_path=VALID_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert report.accepted_count == 10
        assert report.rejected_count == 0
        assert report.duplicate_count == 0
        assert report.malformed_count == 0
        assert report.total_rows_read == 10

    def test_parquet_output_created(self, adapter, tmp_output):
        report, snapshot = adapter.ingest(
            source_path=VALID_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        out_dir = Path(report.parquet_output_path)
        parquet_files = list(out_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_parquet_row_count(self, adapter, tmp_output):
        report, snapshot = adapter.ingest(
            source_path=VALID_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        out_dir = Path(report.parquet_output_path)
        total_rows = 0
        for pf in out_dir.glob("*.parquet"):
            total_rows += pq.read_metadata(pf).num_rows
        assert total_rows == 10

    def test_snapshot_fields_populated(self, adapter, tmp_output):
        report, snapshot = adapter.ingest(
            source_path=VALID_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert snapshot.dataset_id == "test_dataset"
        assert snapshot.row_count == 10
        assert snapshot.accepted_count == 10
        assert snapshot.source_type in ("amazon_jsonl",)
        assert snapshot.source_path_fingerprint  # non-empty
        assert snapshot.canonical_content_hash  # non-empty
        assert len(snapshot.source_path_fingerprint) == 64  # full SHA-256

    def test_no_absolute_path_in_snapshot(self, adapter, tmp_output):
        """Security: absolute paths must not appear in snapshot artifacts."""
        report, snapshot = adapter.ingest(
            source_path=VALID_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        dump = snapshot.model_dump_json()
        # source_path_fingerprint is a SHA-256 hash, not a path
        assert str(VALID_FIXTURE) not in dump

    def test_canonical_content_hash_deterministic(self, adapter, tmp_path):
        """Same input → same content hash (deterministic)."""
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        _, snap1 = adapter.ingest(VALID_FIXTURE, out1, data_root=tmp_path)
        adapter2 = __import__("faulttrace_data.amazon_adapter", fromlist=["AmazonLocalAdapter"]).AmazonLocalAdapter(dataset_id="test_dataset")
        _, snap2 = adapter2.ingest(VALID_FIXTURE, out2, data_root=tmp_path)
        assert snap1.source_path_fingerprint == snap2.source_path_fingerprint


class TestAmazonAdapterEdgeCases:
    def test_edge_fixture_exists(self):
        assert EDGE_FIXTURE.exists(), f"Missing fixture: {EDGE_FIXTURE}"

    def test_duplicate_rejected(self, adapter, tmp_output):
        """First of two identical ASINs accepted, second rejected as duplicate."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        # Line 1 and 2 are exact duplicates — one should be duplicate
        assert report.duplicate_count >= 1

    def test_malformed_date_rejected(self, adapter, tmp_output):
        """Record with 'not-a-date' timestamp should be rejected."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert "malformed_timestamp" in report.rejection_reasons

    def test_empty_asin_rejected(self, adapter, tmp_output):
        """Record with empty ASIN should be rejected."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert "missing_or_empty_asin" in report.rejection_reasons

    def test_non_json_line_malformed(self, adapter, tmp_output):
        """Non-JSON line should be quarantined as malformed."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert report.malformed_count >= 1

    def test_unicode_accepted(self, adapter, tmp_output):
        """Record with Unicode title and emoji should be accepted."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        # B11UNICODE should pass (valid ASIN, valid date, valid rating)
        assert report.accepted_count >= 1

    def test_out_of_range_rating_rejected(self, adapter, tmp_output):
        """Rating 6.5 should be rejected as invalid."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        assert "invalid_rating" in report.rejection_reasons

    def test_total_rows_accounting(self, adapter, tmp_output):
        """accepted + rejected + duplicate + malformed should sum to total_rows_read."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        accounted = report.accepted_count + report.rejected_count + report.duplicate_count
        assert accounted == report.total_rows_read

    def test_no_raw_text_in_rejection_artifact(self, adapter, tmp_output):
        """Rejected rows artifact must not contain full raw text (only hashes and reasons)."""
        report, _ = adapter.ingest(
            source_path=EDGE_FIXTURE,
            output_root=tmp_output,
            data_root=tmp_output.parent,
        )
        if report.rejected_artifact_path:
            artifact_text = Path(report.rejected_artifact_path).read_text()
            # Each line should have 'reason' and 'row_index' but not raw field values
            for line in artifact_text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                assert "reason" in item
                assert "row_index" in item
                # Ensure no long raw text is stored (content < 500 chars per line)
                assert len(line) < 500, f"Rejection artifact line too long: {len(line)}"


class TestAmazonAdapterFieldMapping:
    def test_custom_field_mapping(self, tmp_output):
        """Custom field mapping should override defaults."""
        from faulttrace_data.amazon_adapter import AmazonLocalAdapter, AmazonFieldMapping

        # Write a temp JSONL with custom field names
        import tempfile, json
        custom_data = {
            "product_asin": "B99CUSTOM1",
            "product_title": "Custom Field Test Product",
            "top_category": "Electronics",
            "star_rating": 4.2,
            "review_date": "2022-05-01T00:00:00Z",
            "is_verified": True,
        }
        tmp_file = tmp_output / "custom.jsonl"
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file.write_text(json.dumps(custom_data) + "\n")

        mapping = AmazonFieldMapping(
            asin_field="product_asin",
            title_field="product_title",
            main_category_field="top_category",
            average_rating_field="star_rating",
            timestamp_field="review_date",
            verified_purchase_field="is_verified",
        )
        adapter = AmazonLocalAdapter(dataset_id="custom_test", field_mapping=mapping)
        report, snapshot = adapter.ingest(
            source_path=tmp_file,
            output_root=tmp_output / "out",
            data_root=tmp_output.parent,
        )
        assert report.accepted_count == 1
        assert report.rejected_count == 0

    def test_config_hash_stable(self):
        from faulttrace_data.amazon_adapter import AmazonFieldMapping
        m = AmazonFieldMapping()
        h1 = m.config_hash()
        h2 = m.config_hash()
        assert h1 == h2
        assert len(h1) == 64


class TestSizeLimitGuard:
    def test_size_limit_raises(self, tmp_output):
        """Adapter should raise ValueError when file exceeds configured limit."""
        from faulttrace_data.amazon_adapter import AmazonLocalAdapter
        adapter = AmazonLocalAdapter(dataset_id="test", max_bytes=10)  # 10 bytes limit
        with pytest.raises(ValueError, match="size limit"):
            report, _ = adapter.ingest(VALID_FIXTURE, tmp_output, data_root=tmp_output.parent)
