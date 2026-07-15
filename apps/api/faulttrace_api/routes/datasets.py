"""
Dataset snapshot REST API endpoints — Prompt 2 (WP8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from faulttrace_api.config import get_settings

router = APIRouter()


def _get_registry():
    from faulttrace_data.snapshot import SnapshotRegistry
    settings = get_settings()
    registry_path = settings.data_root / "manifests" / "snapshots.jsonl"
    return SnapshotRegistry(registry_path)


@router.get("/datasets", summary="List all ingested dataset snapshots")
async def list_datasets(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    active_only: bool = Query(True, description="Only return active snapshots"),
) -> dict[str, Any]:
    try:
        registry = _get_registry()
        snapshots = registry.list_snapshots(dataset_id=dataset_id, active_only=active_only)
        return {
            "count": len(snapshots),
            "snapshots": [
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
                    "canonical_content_hash": s.canonical_content_hash[:16] if s.canonical_content_hash else "",
                }
                for s in snapshots
            ],
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
