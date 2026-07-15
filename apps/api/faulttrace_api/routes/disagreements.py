"""
Gold disagreements REST API endpoints — Prompt 2 (WP8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from faulttrace_api.config import get_settings

router = APIRouter()


@router.get("/disagreements", summary="List gold engine disagreements for a world")
async def list_disagreements(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
) -> dict[str, Any]:
    """List disagreement reports from dual gold validation."""
    settings = get_settings()
    reports_dir = settings.data_root.parent / "artifacts" / "gold_reports"

    if not reports_dir.exists():
        return {"count": 0, "disagreements": []}

    import json
    disagreements = []
    pattern = f"disagreements_{world_id}.jsonl" if world_id else "disagreements_*.jsonl"
    for report_path in sorted(reports_dir.glob(pattern)):
        try:
            with open(report_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    disagreements.append({
                        "query_id": item.get("query_id"),
                        "world_id": item.get("world_id"),
                        "template_id": item.get("template_id"),
                        "family": item.get("family"),
                        "pandas_result": item.get("pandas_result"),
                        "duckdb_result": item.get("duckdb_result"),
                        "tolerance": item.get("tolerance"),
                        "computed_at": item.get("computed_at"),
                    })
        except Exception:
            pass

    return {"count": len(disagreements), "disagreements": disagreements}


@router.get("/disagreements/{world_id}/summary", summary="Get disagreement summary for a world")
async def get_disagreement_summary(world_id: str) -> dict[str, Any]:
    """Return aggregated disagreement statistics for a world."""
    settings = get_settings()
    reports_dir = settings.data_root.parent / "artifacts" / "gold_reports"
    report_path = reports_dir / f"validation_report_{world_id}.json"

    if not report_path.exists():
        return {
            "world_id": world_id,
            "status": "no_report",
            "message": "No validation report found. Run 'faulttrace gold validate' first.",
        }

    import json
    try:
        data = json.loads(report_path.read_text())
        return {
            "world_id": world_id,
            "total": data.get("total", 0),
            "agreed": data.get("agreed", 0),
            "disagreed": data.get("disagreed", 0),
            "skipped": data.get("skipped", 0),
            "agreement_rate": (
                data.get("agreed", 0) / max(data.get("total", 1), 1)
            ),
            "by_family": data.get("by_family", {}),
            "generated_at": data.get("generated_at"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
