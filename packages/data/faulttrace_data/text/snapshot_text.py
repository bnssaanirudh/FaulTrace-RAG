"""
Text Corpus Snapshot and Registry.
"""

from typing import Any
from uuid import uuid4

import orjson
from pydantic import BaseModel, Field

from faulttrace_data.snapshot import SnapshotRegistry, _sha256, _utcnow

TEXT_SNAPSHOT_SCHEMA_VERSION = "1.0.0"

class TextCorpusSnapshot(BaseModel):
    """
    Full provenance record for one ingestion run of a text dataset.
    """
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_id: str
    schema_version: str = TEXT_SNAPSHOT_SCHEMA_VERSION

    # Source provenance
    source_type: str = Field(..., description="'jsonl', 'csv', 'txt', 'html', 'pdf', 'mixed'")
    source_path_fingerprint: str = Field(..., description="SHA-256 of relative source path; no absolute path stored")
    source_file_count: int = 0
    source_file_size_bytes: int | None = None

    # Canonical output
    canonical_schema_version: str = TEXT_SNAPSHOT_SCHEMA_VERSION
    document_count: int = 0
    chunk_count: int = 0
    parquet_root: str | None = None  # relative path under data_root

    # Summary statistics
    languages: dict[str, int] = Field(default_factory=dict)
    size_distribution: dict[str, int] = Field(default_factory=dict) # E.g. {"<1KB": 10, "1KB-10KB": 5}
    top_terms: list[str] = Field(default_factory=list)

    # Ingestion Report & Deduplication
    accepted_documents: int = 0
    rejected_documents: int = 0
    duplicate_documents: int = 0
    exact_duplicate_chunks: int = 0
    near_duplicate_chunks: int = 0
    malformed_documents: int = 0
    ingestion_warnings: list[str] = Field(default_factory=list)

    # Hashes
    raw_content_hash: str = Field(default="", description="SHA-256 hash of raw source bytes across files")
    canonical_content_hash: str = Field(default="", description="SHA-256 hash of canonical Parquet bytes")
    ingestion_config_hash: str = Field(default="", description="SHA-256 hash of text ingestion config")

    # Provenance
    license_note: str = Field(default="", description="License/provenance note entered by the researcher")
    producing_command: str | None = None
    creation_tool_version: str = "faulttrace-text-1.0.0"
    environment_summary: dict[str, Any] = Field(default_factory=dict)

    # Status
    active: bool = True
    created_at: str = Field(default_factory=_utcnow)

    def snapshot_hash(self) -> str:
        data = {
            "dataset_id": self.dataset_id,
            "source_path_fingerprint": self.source_path_fingerprint,
            "ingestion_config_hash": self.ingestion_config_hash,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "canonical_content_hash": self.canonical_content_hash,
        }
        return _sha256(orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode())

    @classmethod
    def make_snapshot_id(cls, dataset_id: str, source_hash: str, config_hash: str) -> str:
        raw = f"text:{dataset_id}:{source_hash}:{config_hash}"
        return _sha256(raw)[:16]


class TextSnapshotRegistry(SnapshotRegistry):
    """
    JSONL-based registry of TextCorpusSnapshot records.
    Inherits generic file ops from SnapshotRegistry.
    """
    def _load_all(self) -> list[TextCorpusSnapshot]:
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
                    records.append(TextCorpusSnapshot.model_validate_json(line))
                except Exception:
                    pass
        by_id: dict[str, TextCorpusSnapshot] = {}
        for r in records:
            by_id[r.snapshot_id] = r
        return list(by_id.values())
