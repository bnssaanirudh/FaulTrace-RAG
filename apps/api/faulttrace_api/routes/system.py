"""System status endpoint."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from faulttrace_pipelines import PIPELINE_REGISTRY
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import QueryRow, RunRow, WorldRow, get_db

router = APIRouter()


@router.get("/system/status", summary="System status and component health")
async def system_status(db: Session = Depends(get_db)):
    settings = get_settings()

    world_count = db.query(WorldRow).count()
    query_count = db.query(QueryRow).count()
    run_count = db.query(RunRow).count()

    data_root_exists = settings.data_root.exists()
    artifacts_root_exists = settings.artifacts_root.exists()

    return {
        "status": "operational",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.1.0",
        "milestone": "Prompt 1 - ~30% complete",
        "components": {
            "database": "ok",
            "data_root": "ok" if data_root_exists else "missing",
            "artifacts_root": "ok" if artifacts_root_exists else "missing",
        },
        "counts": {
            "worlds": world_count,
            "queries": query_count,
            "runs": run_count,
        },
        "settings": {
            "data_root": str(settings.data_root),
            "artifacts_root": str(settings.artifacts_root),
            "database_url": "sqlite (default)" if not settings.database_url else "custom",
            "demo_seed": settings.demo_seed,
        },
        "pipelines": dict.fromkeys(PIPELINE_REGISTRY, "available"),
        "milestone": "Prompt 2 — P1-P5 + Attribution Engine",
    }
