"""
Dataset snapshot REST API endpoints — Prompt 2 (WP8).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from faulttrace_api.config import get_settings

router = APIRouter()


def _resolve_trusted_input(input_path: str, *, allow_directory: bool = False) -> Path:
    """Resolve an ingestion path and enforce explicit trusted-root containment."""
    settings = get_settings()
    try:
        resolved = Path(input_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or non-existent path") from exc

    configured = [Path(value).resolve() for value in settings.trusted_ingest_roots.split(os.pathsep) if value]
    fixture_root = Path("apps/api/tests/fixtures").resolve()
    roots = [settings.data_root.resolve(), *configured]
    if fixture_root.exists():
        roots.append(fixture_root)
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=400, detail="Path traversal detected: outside trusted roots")
    if resolved.is_dir() and not allow_directory:
        raise HTTPException(status_code=400, detail="Expected a file, not a directory")
    return resolved


def _get_registry():
    from faulttrace_data.snapshot import SnapshotRegistry

    settings = get_settings()
    registry_path = settings.data_root / "manifests" / "snapshots.jsonl"
    return SnapshotRegistry(registry_path)


@router.get("/datasets", summary="List all ingested dataset snapshots")
async def list_datasets(
    dataset_id: str | None = Query(None, description="Filter by dataset ID"),
    active_only: bool = Query(True, description="Only return active snapshots"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    try:
        registry = _get_registry()
        snapshots = registry.list_snapshots(dataset_id=dataset_id, active_only=active_only)

        total = len(snapshots)
        start = (page - 1) * page_size
        end = start + page_size
        page_snapshots = snapshots[start:end]

        return {
            "items": [
                {
                    "snapshot_id": s.snapshot_id,
                    "dataset_id": s.dataset_id,
                    "source_type": s.source_type,
                    "row_count": s.row_count,
                    "accepted_count": s.accepted_count,
                    "rejected_count": s.rejected_count,
                    "duplicate_count": s.duplicate_count,
                    "malformed_count": s.malformed_count,
                    "active": s.active,
                    "created_at": s.created_at,
                    "license_note": s.license_note,
                    "canonical_content_hash": s.canonical_content_hash[:16]
                    if s.canonical_content_hash
                    else "",
                }
                for s in page_snapshots
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": end < total,
            "count": len(page_snapshots),  # For backwards compatibility
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{snapshot_id}", summary="Get a specific snapshot by ID")
async def get_snapshot(snapshot_id: str) -> dict[str, Any]:
    try:
        registry = _get_registry()
        snapshot = registry.inspect(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
        return snapshot.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{snapshot_id}/validate", summary="Validate a snapshot's integrity")
async def validate_snapshot(snapshot_id: str) -> dict[str, Any]:
    try:
        settings = get_settings()
        registry = _get_registry()
        result = registry.validate(snapshot_id, data_root=settings.data_root)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{snapshot_id}", summary="Deactivate a snapshot (tombstone)")
async def deactivate_snapshot(snapshot_id: str) -> dict[str, Any]:
    try:
        registry = _get_registry()
        found = registry.deactivate(snapshot_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
        return {"status": "deactivated", "snapshot_id": snapshot_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{snapshot_id}/missingness", summary="Get missingness summary for a snapshot")
async def get_missingness(snapshot_id: str) -> dict[str, Any]:
    try:
        registry = _get_registry()
        snapshot = registry.inspect(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
        return {
            "snapshot_id": snapshot_id,
            "missingness": snapshot.missingness.model_dump(),
            "null_count_by_field": snapshot.null_count_by_field,
            "row_count": snapshot.row_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


class IngestRequest(BaseModel):
    input_path: str
    dataset_id: str
    license_note: str = ""
    max_bytes_mb: int = 500


@router.post("/datasets/ingest", summary="Ingest a local Amazon-style file as a snapshot")
async def ingest_dataset(request: IngestRequest) -> dict[str, Any]:
    from faulttrace_data.amazon_adapter import AmazonLocalAdapter
    from faulttrace_data.snapshot import SnapshotRegistry

    settings = get_settings()
    resolved_path = _resolve_trusted_input(request.input_path)

    # SECURITY: Extension / MIME check (simple)
    if resolved_path.suffix.lower() not in {".json", ".jsonl"}:
        raise HTTPException(status_code=400, detail="Only .json or .jsonl files are allowed")

    output = settings.data_root / "snapshots"
    data_root = settings.data_root

    adapter = AmazonLocalAdapter(
        dataset_id=request.dataset_id,
        max_bytes=request.max_bytes_mb * 1024 * 1024,
    )

    producing_cmd = f"api: POST /api/v1/datasets/ingest --input {resolved_path.name}"

    try:
        report, snapshot = adapter.ingest(
            source_path=resolved_path,
            output_root=output,
            data_root=data_root,
            license_note=request.license_note,
            producing_command=producing_cmd,
        )
        registry_path = data_root / "manifests" / "snapshots.jsonl"
        registry = SnapshotRegistry(registry_path)
        registry.register(snapshot)
        return {
            "status": "success",
            "snapshot_id": snapshot.snapshot_id,
            "total_rows_read": report.total_rows_read,
            "accepted_count": report.accepted_count,
            "rejected_count": report.rejected_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Text Corpus Endpoints
# ---------------------------------------------------------------------------


class TextIngestRequest(BaseModel):
    input_path: str
    dataset_id: str
    source_type: str
    license_note: str = ""
    chunk_size: int = 1000
    overlap: int = 100
    strict_chunk_dedup: bool = False
    text_field: str = "text"
    id_field: str = "id"
    title_field: str = "title"


def _get_text_registry():
    from faulttrace_data.text.snapshot_text import TextSnapshotRegistry

    settings = get_settings()
    registry_path = settings.data_root / "manifests" / "text_snapshots.jsonl"
    return TextSnapshotRegistry(registry_path)


@router.post("/datasets/text/ingest", summary="Ingest a text corpus")
async def ingest_text_dataset(request: TextIngestRequest) -> dict[str, Any]:
    from faulttrace_data.text.pipeline import TextIngestionPipeline

    settings = get_settings()
    resolved_path = _resolve_trusted_input(request.input_path, allow_directory=True)

    registry_path = settings.data_root / "manifests" / "text_snapshots.jsonl"
    pipeline = TextIngestionPipeline(data_root=settings.data_root, registry_path=registry_path)

    try:
        snapshot = pipeline.run(
            dataset_id=request.dataset_id,
            source_path=resolved_path,
            source_type=request.source_type,
            config={
                "chunk_size": request.chunk_size,
                "overlap": request.overlap,
                "strict_chunk_dedup": request.strict_chunk_dedup,
                "text_field": request.text_field,
                "id_field": request.id_field,
                "title_field": request.title_field,
            },
        )
        return {
            "status": "success",
            "snapshot_id": snapshot.snapshot_id,
            "document_count": snapshot.document_count,
            "chunk_count": snapshot.chunk_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/text", summary="List all ingested text dataset snapshots")
async def list_text_datasets(
    dataset_id: str | None = Query(None, description="Filter by dataset ID"),
    active_only: bool = Query(True, description="Only return active snapshots"),
) -> dict[str, Any]:
    try:
        registry = _get_text_registry()
        snapshots = registry.list_snapshots(dataset_id=dataset_id, active_only=active_only)

        return {
            "items": [
                {
                    "snapshot_id": s.snapshot_id,
                    "dataset_id": s.dataset_id,
                    "source_type": s.source_type,
                    "document_count": s.document_count,
                    "chunk_count": s.chunk_count,
                    "active": s.active,
                    "created_at": s.created_at,
                }
                for s in snapshots
            ],
            "total": len(snapshots),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/text/{snapshot_id}", summary="Get a specific text snapshot by ID")
async def get_text_snapshot(snapshot_id: str) -> dict[str, Any]:
    try:
        registry = _get_text_registry()
        snapshot = registry.inspect(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Text snapshot '{snapshot_id}' not found")
        return snapshot.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/text/{snapshot_id}/preview", summary="Preview chunks of a text snapshot")
async def preview_text_snapshot(
    snapshot_id: str, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)
) -> dict[str, Any]:
    import pandas as pd

    try:
        registry = _get_text_registry()
        snapshot = registry.inspect(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"Text snapshot '{snapshot_id}' not found")

        settings = get_settings()
        parquet_path = settings.data_root / snapshot.parquet_root / "chunks.parquet"

        if not parquet_path.exists():
            raise HTTPException(status_code=404, detail="Parquet chunks not found for snapshot")

        # We read the parquet file to get the preview.
        # This is safe because we only read from generated parquet files and never arbitrary files.
        df = pd.read_parquet(parquet_path)

        total = len(df)
        start = (page - 1) * page_size
        end = min(start + page_size, total)

        preview_df = df.iloc[start:end]
        chunks = preview_df.to_dict(orient="records")

        return {
            "snapshot_id": snapshot_id,
            "total_chunks": total,
            "page": page,
            "page_size": page_size,
            "chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
