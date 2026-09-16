"""Immutable provenance manifests for externally acquired benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkFile(BaseModel):
    split: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class BenchmarkManifest(BaseModel):
    schema_version: str = "1.0.0"
    dataset_id: str
    source_url: str
    source_revision: str | None = None
    acquisition_method: str = "not_recorded"
    provenance_status: str = "locally_hashed_source_snapshot"
    license_name: str
    license_url: str | None = None
    files: list[BenchmarkFile]
    snapshot_sha256: str
    query_split_overlap: dict[str, list[str]] = Field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_query_split_overlap(query_ids_by_split: dict[str, set[str]]) -> dict[str, list[str]]:
    """Return query IDs occurring in more than one declared split."""
    memberships: dict[str, list[str]] = {}
    for split, query_ids in query_ids_by_split.items():
        for query_id in query_ids:
            memberships.setdefault(str(query_id), []).append(split)
    return {
        query_id: sorted(splits)
        for query_id, splits in memberships.items()
        if len(set(splits)) > 1
    }


def build_benchmark_manifest(
    *,
    dataset_id: str,
    root: Path,
    split_files: dict[str, list[Path]],
    source_url: str,
    source_revision: str | None = None,
    acquisition_method: str = "not_recorded",
    provenance_status: str = "locally_hashed_source_snapshot",
    license_name: str,
    license_url: str | None = None,
    query_ids_by_split: dict[str, set[str]] | None = None,
) -> BenchmarkManifest:
    """Hash an explicit file set; implicit directory discovery is intentionally forbidden."""
    root = root.resolve()
    files: list[BenchmarkFile] = []
    for split in sorted(split_files):
        for candidate in sorted(split_files[split], key=lambda path: path.as_posix()):
            path = candidate.resolve()
            if not path.is_file() or not path.is_relative_to(root):
                raise ValueError(f"Benchmark file is missing or outside root: {candidate}")
            files.append(
                BenchmarkFile(
                    split=split,
                    relative_path=path.relative_to(root).as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=_sha256(path),
                )
            )
    if not files:
        raise ValueError("At least one benchmark file is required")
    canonical = json.dumps(
        [file.model_dump(mode="json") for file in files],
        sort_keys=True,
        separators=(",", ":"),
    )
    overlap = find_query_split_overlap(query_ids_by_split or {})
    return BenchmarkManifest(
        dataset_id=dataset_id,
        source_url=source_url,
        source_revision=source_revision,
        acquisition_method=acquisition_method,
        provenance_status=provenance_status,
        license_name=license_name,
        license_url=license_url,
        files=files,
        snapshot_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        query_split_overlap=overlap,
    )
