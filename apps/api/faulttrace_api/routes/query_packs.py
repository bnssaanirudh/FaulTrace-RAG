"""
Query packs REST API endpoints — Prompt 2 (WP8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from faulttrace_api.config import get_settings

router = APIRouter()


@router.get("/query-packs", summary="List available query benchmark packs")
async def list_packs(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
) -> dict[str, Any]:
    """List all benchmark pack JSON files in the artifacts/query_packs directory."""
    settings = get_settings()
    packs_dir = settings.data_root.parent / "artifacts" / "query_packs"

    if not packs_dir.exists():
        return {"count": 0, "packs": []}

    import json
    packs = []
    for pack_path in sorted(packs_dir.glob("pack_*.json")):
        try:
            data = json.loads(pack_path.read_text())
            if world_id and data.get("world_id") != world_id:
                continue
            packs.append({
                "pack_id": data.get("pack_id"),
                "world_id": data.get("world_id"),
                "total_count": data.get("total_count"),
                "agreed_count": data.get("agreed_count"),
                "disagreed_count": data.get("disagreed_count"),
                "gold_ready": data.get("gold_ready"),
                "dev_count": data.get("dev_count"),
                "val_count": data.get("val_count"),
                "test_count": data.get("test_count"),
                "created_at": data.get("created_at"),
                "file": pack_path.name,
            })
        except Exception:
            pass

    return {"count": len(packs), "packs": packs}


@router.get("/query-packs/{world_id}", summary="Get a benchmark pack for a world")
async def get_pack(world_id: str) -> dict[str, Any]:
    import json
    settings = get_settings()
    packs_dir = settings.data_root.parent / "artifacts" / "query_packs"
    pack_path = packs_dir / f"pack_{world_id}.json"

    if not pack_path.exists():
        raise HTTPException(status_code=404, detail=f"Pack for world '{world_id}' not found")

    try:
        return json.loads(pack_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-packs/{world_id}/distribution", summary="Get query distribution for a pack")
async def get_pack_distribution(world_id: str) -> dict[str, Any]:
    import json
    settings = get_settings()
    packs_dir = settings.data_root.parent / "artifacts" / "query_packs"
    pack_path = packs_dir / f"pack_{world_id}.json"

    if not pack_path.exists():
        raise HTTPException(status_code=404, detail=f"Pack for world '{world_id}' not found")

    try:
        data = json.loads(pack_path.read_text())
        return {
            "world_id": world_id,
            "total": data.get("total_count"),
            "by_family": data.get("count_by_family", {}),
            "by_difficulty": data.get("count_by_difficulty", {}),
            "splits": {
                "dev": data.get("dev_count"),
                "val": data.get("val_count"),
                "test": data.get("test_count"),
            },
            "gold_ready": data.get("gold_ready"),
            "agreed_count": data.get("agreed_count"),
            "disagreed_count": data.get("disagreed_count"),
            "duplicate_spec_hashes_found": data.get("duplicate_spec_hashes_found", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template-registry", summary="List all available query templates")
async def list_templates() -> dict[str, Any]:
    try:
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        entries = []
        for e in TEMPLATE_REGISTRY._entries.values():
            entries.append({
                "template_id": e.template_id,
                "family": e.family,
                "difficulty": e.difficulty,
                "selectivity": e.selectivity,
                "null_risk": e.null_risk,
                "tie_risk": e.tie_risk,
                "temporal_risk": e.temporal_risk,
            })
        summary = TEMPLATE_REGISTRY.summary()
        return {
            "summary": summary,
            "templates": entries,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
