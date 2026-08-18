"""
FastAPI router for controlled reproducibility experiments and matrix execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from faulttrace_reporting import (
    ExperimentSpec,
    MetricsComputer,
    ResumableMatrixRunner,
    compute_paired_bootstrap_ci,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from faulttrace_api.database import ExperimentRow, RunRow, get_db

router = APIRouter()
logger = logging.getLogger("faulttrace.api.experiments")

# Simple active cancelled-flags map
_running_states: dict[str, str] = {}


class CompareRequest(BaseModel):
    experiment_id_1: str
    experiment_id_2: str


@router.post("/experiments/plan", summary="Plan matrix execution and estimate API cost")
async def plan_experiment(spec: ExperimentSpec):
    try:
        runner = ResumableMatrixRunner(spec)
        plan = runner.dry_run()

        if plan.get("total_jobs", 0) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Matrix size exceeds the 10,000 job limit for security and budget constraints.",
            )

        return plan
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/experiments/run", summary="Trigger or resume background matrix execution")
async def run_experiment(
    spec: ExperimentSpec, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    try:
        runner = ResumableMatrixRunner(spec, db)
        config_hash = runner.config_hash

        # Security: Enforce hard cap
        plan = runner.dry_run()
        if plan.get("total_jobs", 0) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Matrix size exceeds the 10,000 job limit for security and budget constraints.",
            )

        # Check if already running
        if _running_states.get(config_hash) == "running":
            return {"status": "already_running", "experiment_id": config_hash}

        _running_states[config_hash] = "running"

        def task_wrapper():
            try:
                runner.run(
                    cancel_flag_callback=lambda: _running_states.get(config_hash) == "cancelled"
                )
            finally:
                _running_states.pop(config_hash, None)

        background_tasks.add_task(task_wrapper)
        return {"status": "started", "experiment_id": config_hash}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/experiments", summary="List all experiments and their statuses")
async def list_experiments(db: Session = Depends(get_db)):
    rows = db.query(ExperimentRow).order_by(ExperimentRow.created_at.desc()).all()
    results = []
    for r in rows:
        results.append(
            {
                "experiment_id": r.experiment_id,
                "name": r.name,
                "status": r.status,
                "total_jobs": r.total_jobs,
                "completed_jobs": r.completed_jobs,
                "failed_jobs": r.failed_jobs,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
            }
        )
    return {"experiments": results}


@router.get("/experiments/{id}", summary="Get details and progress for an experiment")
async def get_experiment(id: str, db: Session = Depends(get_db)):
    row = db.query(ExperimentRow).filter(ExperimentRow.experiment_id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Calculate detailed performance metrics if complete
    runs = db.query(RunRow).filter(RunRow.experiment_id == id).all()
    runs_list = []
    for r in runs:
        runs_list.append(
            {
                "run_id": r.run_id,
                "is_correct": r.is_correct,
                "loss": r.loss,
                "latency_ms": r.latency_ms,
                "pipeline_id": r.pipeline_id,
                "provider_id": r.provider_id,
                "policy_decision": r.policy_decision,
                "answer": r.answer,
                "status": r.status,
            }
        )

    metrics = MetricsComputer.compute_all(runs_list)

    return {
        "experiment_id": row.experiment_id,
        "name": row.name,
        "status": row.status,
        "config": json.loads(row.config_json),
        "total_jobs": row.total_jobs,
        "completed_jobs": row.completed_jobs,
        "failed_jobs": row.failed_jobs,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
        "metrics": metrics.model_dump(),
    }


@router.post("/experiments/{id}/cancel", summary="Cancel a running experiment")
async def cancel_experiment(id: str, db: Session = Depends(get_db)):
    row = db.query(ExperimentRow).filter(ExperimentRow.experiment_id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if id in _running_states:
        _running_states[id] = "cancelled"

    row.status = "cancelled"
    db.commit()
    return {"status": "cancelled", "experiment_id": id}


@router.post(
    "/experiments/compare",
    summary="Compare two experiments and calculate bootstrap confidence intervals",
)
async def compare_experiments(request: CompareRequest, db: Session = Depends(get_db)):
    id1 = request.experiment_id_1
    id2 = request.experiment_id_2

    runs1 = db.query(RunRow).filter(RunRow.experiment_id == id1, RunRow.status == "completed").all()
    runs2 = db.query(RunRow).filter(RunRow.experiment_id == id2, RunRow.status == "completed").all()

    if not runs1 or not runs2:
        raise HTTPException(
            status_code=422, detail="Both experiments must have completed run outputs to compare."
        )

    # Match by query_id and pipeline_id to perform query-level pairing
    runs2_map = {(r.query_id, r.pipeline_id): r for r in runs2}

    paired_acc1 = []
    paired_acc2 = []
    paired_loss1 = []
    paired_loss2 = []

    for r1 in runs1:
        key = (r1.query_id, r1.pipeline_id)
        if key in runs2_map:
            r2 = runs2_map[key]
            paired_acc1.append(1.0 if r1.is_correct else 0.0)
            paired_acc2.append(1.0 if r2.is_correct else 0.0)
            paired_loss1.append(float(r1.loss or 0))
            paired_loss2.append(float(r2.loss or 0))

    if len(paired_acc1) < 2:
        raise HTTPException(
            status_code=422, detail="Insufficient paired matching query runs to compute bootstrap."
        )

    # Compute bootstrap CI on accuracy
    diff_acc, ci_acc, effect_acc = compute_paired_bootstrap_ci(paired_acc1, paired_acc2)
    # Compute bootstrap CI on error loss
    diff_loss, ci_loss, effect_loss = compute_paired_bootstrap_ci(paired_loss1, paired_loss2)

    return {
        "paired_samples": len(paired_acc1),
        "accuracy": {
            "mean_diff": diff_acc,
            "confidence_interval": ci_acc,
            "cohens_d": effect_acc,
        },
        "loss": {
            "mean_diff": diff_loss,
            "confidence_interval": ci_loss,
            "cohens_d": effect_loss,
        },
    }


@router.get("/experiments/{id}/download/{filename:path}", summary="Download artifact file")
async def download_experiment_artifact(id: str, filename: str, db: Session = Depends(get_db)):
    row = db.query(ExperimentRow).filter(ExperimentRow.experiment_id == id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")

    config = json.loads(row.config_json)
    output_root = Path(config.get("output_root", "artifacts/experiments"))
    file_path = output_root / id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(path=str(file_path), filename=file_path.name)
