"""Artifacts metadata endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import RunRow, get_db

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
            try:
                relative_path = str(p.resolve().relative_to(settings.artifacts_root.resolve()))
            except (OSError, ValueError):
                relative_path = None
            sha256 = None
            if p.exists():
                hasher = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                sha256 = hasher.hexdigest()

            artifact_info[name] = {
                "relative_path": relative_path,
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "sha256": sha256,
            }
        return {
            "artifact_id": artifact_id,
            "type": "run_artifacts",
            "run_id": run.run_id,
            "artifacts": artifact_info,
        }

    raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")
