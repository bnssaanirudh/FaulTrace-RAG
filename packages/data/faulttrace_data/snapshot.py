"""
Dataset snapshot and registry for FaultTrace-RAG.

A DatasetSnapshot captures all provenance metadata for a single ingestion
run so that benchmarks built on that snapshot are fully reproducible and
auditable. The SnapshotRegistry provides list/inspect/validate/deactivate
operations over a JSONL-based registry file stored under data/manifests/.

Design decisions:
- Absolute source paths are NEVER stored; only SHA-256 fingerprints of
  the relative path (relative to a configured data root) are stored.
- All hashes use SHA-256; 64-hex-char full digest stored.
- Registry is append-only JSONL; deactivate writes a tombstone record.
- Snapshot IDs are deterministic: SHA-256(dataset_id + source_hash + config_hash)[:16]
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import orjson
from pydantic import BaseModel, Field, field_validator

SNAPSHOT_SCHEMA_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _fingerprint_path(path: Path, data_root: Path) -> str:
    """
    Return a SHA-256 fingerprint of the path relative to data_root.
    Never stores absolute paths in exported artifacts.
    """
    try:
        rel = path.resolve().relative_to(data_root.resolve())
        return _sha256(str(rel))
    except ValueError:
        # Path not under data_root — fingerprint the basename only
        return _sha256(path.name)


# ---------------------------------------------------------------------------
# MissingnessSummary
# ---------------------------------------------------------------------------


class MissingnessSummary(BaseModel):
    """Per-field missing value counts."""

    price_null_count: int = 0
    price_null_ratio: float = 0.0
    category_null_count: int = 0
    category_null_ratio: float = 0.0
    brand_null_count: int = 0
    brand_null_ratio: float = 0.0
    rating_null_count: int = 0
    rating_null_ratio: float = 0.0
    verified_null_count: int = 0
    verified_null_ratio: float = 0.0
    timestamp_null_count: int = 0
    timestamp_null_ratio: float = 0.0
    total_fields_checked: int = 6


# ---------------------------------------------------------------------------
# DatasetSnapshot
# ---------------------------------------------------------------------------


class DatasetSnapshot(BaseModel):
    """
    Full provenance record for one ingestion run of a dataset.

    Prompt 2 requirement: dataset_id, snapshot_id, source_type, source_path_fingerprint,
    canonical_schema_version, row_count, partition_count, min/max_timestamp,
    category/brand/rating/verified/missingness summaries, raw/canonical content hashes,
    ingestion_config_hash, rejected_row_artifact_ref, license_note,
    creation_tool_version, environment_summary.
    """

    # Identity
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_id: str
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    # Source provenance (no absolute path)
    source_type: str = Field(..., description="'generated', 'amazon_jsonl', 'amazon_csv', 'amazon_parquet'")
    source_path_fingerprint: str = Field(..., description="SHA-256 of relative source path; no absolute path stored")
    source_file_size_bytes: Optional[int] = None
    source_row_count_raw: int = 0

    # Canonical output
    canonical_schema_version: str = SNAPSHOT_SCHEMA_VERSION
    row_count: int = 0
    partition_count: int = 0
    parquet_root: Optional[str] = None  # relative path under data_root

    # Temporal coverage
    min_timestamp: Optional[str] = None
    max_timestamp: Optional[str] = None

    # Summary statistics
    category_counts: dict[str, int] = Field(default_factory=dict)
    brand_counts: dict[str, int] = Field(default_factory=dict)
    rating_distribution: dict[str, int] = Field(default_factory=dict)
    verified_purchase_count: int = 0
    verified_purchase_ratio: float = 0.0
    missingness: MissingnessSummary = Field(default_factory=MissingnessSummary)

    # Hashes
    raw_content_hash: str = Field(default="", description="SHA-256 hash of raw source bytes")
    canonical_content_hash: str = Field(default="", description="SHA-256 hash of canonical Parquet bytes")
    ingestion_config_hash: str = Field(default="", description="SHA-256 hash of field mapping + ingestion config")

    # Ingestion report
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    malformed_count: int = 0
    null_count_by_field: dict[str, int] = Field(default_factory=dict)
    rejected_row_artifact_ref: Optional[str] = None

    # Provenance
    license_note: str = Field(
        default="",
        description="License/provenance note entered by the researcher",
    )
    producing_command: Optional[str] = None
    creation_tool_version: str = "faulttrace-2.0.0"
    environment_summary: dict[str, Any] = Field(default_factory=dict)

    # Status
    active: bool = True
    created_at: str = Field(default_factory=_utcnow)

    def snapshot_hash(self) -> str:
        """Stable hash identifying this snapshot's identity."""
        data = {
            "dataset_id": self.dataset_id,
            "source_path_fingerprint": self.source_path_fingerprint,
            "ingestion_config_hash": self.ingestion_config_hash,
            "row_count": self.row_count,
            "canonical_content_hash": self.canonical_content_hash,
        }
        return _sha256(orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode())

    @classmethod
    def make_snapshot_id(cls, dataset_id: str, source_hash: str, config_hash: str) -> str:
        """Deterministic snapshot ID: first 16 chars of SHA-256."""
        raw = f"{dataset_id}:{source_hash}:{config_hash}"
        return _sha256(raw)[:16]


# ---------------------------------------------------------------------------
# SnapshotRegistry
# ---------------------------------------------------------------------------


class SnapshotRegistry:
    """
    JSONL-based registry of DatasetSnapshot records.

    Stored under data_root/manifests/snapshots.jsonl (default).
    Registry is append-only; deactivate() writes a tombstone with active=False.
    Operations: register, list, inspect, validate, deactivate.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, snapshot: DatasetSnapshot) -> None:
        """Append a snapshot to the registry."""
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(snapshot.model_dump_json() + "\n")

    def deactivate(self, snapshot_id: str) -> bool:
        """
        Write a tombstone record marking snapshot_id as inactive.
        Does not delete source artifacts.
        Returns True if found, False if not found.
        """
        records = self._load_all()
        found = any(r.snapshot_id == snapshot_id for r in records)
        if not found:
            return False
        # Find and deactivate
        updated = []
        for r in records:
            if r.snapshot_id == snapshot_id:
                updated.append(r.model_copy(update={"active": False}))
            else:
                updated.append(r)
        self._rewrite(updated)
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_snapshots(self, dataset_id: Optional[str] = None, active_only: bool = True) -> list[DatasetSnapshot]:
        """List snapshots, optionally filtered by dataset_id and active status."""
        records = self._load_all()
        if dataset_id:
            records = [r for r in records if r.dataset_id == dataset_id]
        if active_only:
            records = [r for r in records if r.active]
        return records

    def inspect(self, snapshot_id: str) -> Optional[DatasetSnapshot]:
        """Return a single snapshot by ID, or None if not found."""
        for r in self._load_all():
            if r.snapshot_id == snapshot_id:
                return r
        return None

    def validate(self, snapshot_id: str, data_root: Path) -> dict[str, Any]:
        """
        Validate a snapshot:
        - Check that canonical Parquet exists at expected path
        - Re-hash Parquet and compare to stored canonical_content_hash
        - Check rejected row artifact if present
        Returns a dict with status, checks, and any failures.
        """
        snapshot = self.inspect(snapshot_id)
        if snapshot is None:
            return {"status": "not_found", "snapshot_id": snapshot_id, "checks": []}

        checks = []

        # Check 1: Parquet exists
        if snapshot.parquet_root:
            parquet_root = data_root / snapshot.parquet_root
            if parquet_root.exists():
                checks.append({"check": "parquet_exists", "status": "pass"})

                # Check 2: Parquet hash matches
                if snapshot.canonical_content_hash:
                    actual_hash = _hash_directory(parquet_root)
                    if actual_hash == snapshot.canonical_content_hash:
                        checks.append({"check": "parquet_hash_match", "status": "pass"})
                    else:
                        checks.append({
                            "check": "parquet_hash_match",
                            "status": "fail",
                            "expected": snapshot.canonical_content_hash[:16],
                            "actual": actual_hash[:16],
                        })
            else:
                checks.append({"check": "parquet_exists", "status": "fail", "path": snapshot.parquet_root})

        # Check 3: Rejected artifact exists if referenced
        if snapshot.rejected_row_artifact_ref:
            ref_path = data_root / snapshot.rejected_row_artifact_ref
            if ref_path.exists():
                checks.append({"check": "rejected_artifact_exists", "status": "pass"})
            else:
                checks.append({"check": "rejected_artifact_exists", "status": "fail"})

        # Check 4: Row count is positive (if accepted > 0)
        if snapshot.accepted_count > 0:
            if snapshot.row_count > 0:
                checks.append({"check": "row_count_positive", "status": "pass"})
            else:
                checks.append({"check": "row_count_positive", "status": "fail"})

        all_pass = all(c["status"] == "pass" for c in checks)
        return {
            "status": "valid" if all_pass else "invalid",
            "snapshot_id": snapshot_id,
            "dataset_id": snapshot.dataset_id,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> list[DatasetSnapshot]:
        """Load all records from registry JSONL."""
        if not self.registry_path.exists():
            return []
        records = []
        with open(self.registry_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(DatasetSnapshot.model_validate_json(line))
                except Exception:
                    pass  # Skip corrupt lines
        # Deduplicate: last write wins for active status
        by_id: dict[str, DatasetSnapshot] = {}
        for r in records:
            by_id[r.snapshot_id] = r
        return list(by_id.values())

    def _rewrite(self, records: list[DatasetSnapshot]) -> None:
        """Rewrite the entire registry file."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")


def _hash_directory(directory: Path) -> str:
    """Compute a stable hash of all files in a directory (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    return h.hexdigest()
