"""Artifacts metadata endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import get_db, RunRow

router = APIRouter()


@router.get("/artifacts/{artifact_id}/metadata", summary="Get artifact metadata")
async def get_artifact_metadata(artifact_id: str, db: Session = Depends(get_db)):
    """Get metadata for a run artifact by run_id."""
    settings = get_settings()
    
    # Check if it's a run artifact
    run = db.query(RunRow).filter(RunRow.run_id == artifact_id).first()
    if run:
        refs = json.loads(run.artifact_refs_json) if run.artifact_refs_json else {}
        artifact_info = {}
        for name, path_str in refs.items():
            p = Path(path_str)
            artifact_info[name] = {
                "path": path_str,
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
            }
        return {
            "artifact_id": artifact_id,
            "type": "run_artifacts",
            "run_id": run.run_id,
            "artifacts": artifact_info,
        }
    
    raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")
