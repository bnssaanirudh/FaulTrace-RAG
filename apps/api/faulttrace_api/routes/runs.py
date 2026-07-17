"""Pipeline runs endpoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import get_db, QueryRow, RunRow, TraceEventRow
from faulttrace_api.models import CreateRunRequest, PaginatedResponse, RunResponse, TraceEventResponse

router = APIRouter()


@router.post("/runs", response_model=RunResponse, summary="Execute a pipeline run")
async def create_run(
    request: CreateRunRequest,
    db: Session = Depends(get_db),
):
    """Execute a pipeline against a query and store the run."""
    settings = get_settings()
    
    # Load query
    q_row = db.query(QueryRow).filter(QueryRow.query_id == request.query_id).first()
    if not q_row:
        raise HTTPException(status_code=404, detail=f"Query '{request.query_id}' not found")
    
    from faulttrace_core.models import GoldAnswer, QuerySpec
    from faulttrace_pipelines import PIPELINE_REGISTRY, get_pipeline
    
    if request.pipeline_id not in PIPELINE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{request.pipeline_id}' not available. "
                   f"Choose from: {list(PIPELINE_REGISTRY)}",
        )
    
    # Parse query spec and gold
    query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
    gold = None
    if q_row.gold_json:
        gold = GoldAnswer.model_validate(json.loads(q_row.gold_json))
    
    # Load corpus data
    world_dir = settings.data_root / "generated" / "worlds" / query_spec.world_id
    parquet_path = world_dir / "records.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"World data not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Run pipeline
    artifacts_dir = settings.artifacts_root / "runs"
    pipeline = get_pipeline(request.pipeline_id, artifacts_dir=artifacts_dir)
    
    try:
        run, events, components = pipeline.run(
            query=query_spec,
            df=df,
            gold_answer=gold,
            parquet_path=parquet_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")
    
    # Store run in DB
    run_row = RunRow(
        run_id=run.run_id,
        query_id=run.query_id,
        pipeline_id=run.pipeline_id,
        provider_id=run.provider_id,
        status=run.status.value,
        answer=str(run.answer) if run.answer is not None else None,
        gold_answer_value=str(run.gold_answer_value) if run.gold_answer_value is not None else None,
        is_correct=run.is_correct,
        loss=run.loss,
        latency_ms=run.latency_ms,
        error_message=run.error_message,
        config_hash=run.config_hash,
        artifact_refs_json=json.dumps(run.artifact_references),
        started_at=run.started_at.replace(tzinfo=None),
        completed_at=run.completed_at.replace(tzinfo=None) if run.completed_at else None,
    )
    db.add(run_row)
    
    # Store trace events
    for ev in events:
        ev_row = TraceEventRow(
            event_id=ev.event_id,
            run_id=ev.run_id,
            parent_event_id=ev.parent_event_id,
            stage=ev.stage,
            event_type=ev.event_type.value,
            message=ev.message,
            record_count_in=ev.record_count_in,
            record_count_out=ev.record_count_out,
            duration_ms=ev.duration_ms,
            payload_json=json.dumps(ev.structured_payload, default=str),
            timestamp=ev.timestamp.replace(tzinfo=None),
        )
        db.add(ev_row)
    
    db.commit()
    
    return _run_row_to_response(run_row)


@router.post("/runs/map-plan", summary="Dry-run map plan")
async def create_map_plan(
    request: CreateRunRequest,
    batch_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Generate an extraction plan without executing the pipeline."""
    settings = get_settings()
    
    q_row = db.query(QueryRow).filter(QueryRow.query_id == request.query_id).first()
    if not q_row:
        raise HTTPException(status_code=404, detail="Query not found")
        
    from faulttrace_core.models import QuerySpec
    from faulttrace_pipelines.scope_service import ScopeService
    from faulttrace_pipelines.planner import MapPlanner
    
    query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
    
    world_dir = settings.data_root / "generated" / "worlds" / query_spec.world_id
    parquet_path = world_dir / "records.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="World data not found")
        
    df = pd.read_parquet(parquet_path)
    scope_res = ScopeService.enumerate_scope(query_spec, df)
    
    if not scope_res.is_success:
        raise HTTPException(status_code=400, detail=f"Scope enumeration failed: {scope_res.failure_code}")
        
    plan = MapPlanner.create_plan("dry-run", query_spec.world_id, scope_res, batch_size=batch_size)
    return plan.model_dump()


@router.get("/runs", response_model=PaginatedResponse[RunResponse], summary="List pipeline runs")
async def list_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    query_id: Optional[str] = Query(None),
    pipeline_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(RunRow).order_by(RunRow.started_at.desc())
    if query_id:
        q = q.filter(RunRow.query_id == query_id)
    if pipeline_id:
        q = q.filter(RunRow.pipeline_id == pipeline_id)
    if status:
        q = q.filter(RunRow.status == status)
    
    total = q.count()
    rows = q.offset((page - 1) * page_size).limit(page_size).all()
    
    return PaginatedResponse(
        items=[_run_row_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get("/runs/{run_id}", response_model=RunResponse, summary="Get run details")
async def get_run(run_id: str, db: Session = Depends(get_db)):
    row = db.query(RunRow).filter(RunRow.run_id == run_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _run_row_to_response(row)


@router.get(
    "/runs/{run_id}/trace",
    response_model=list[TraceEventResponse],
    summary="Get trace events for a run",
)
async def get_run_trace(run_id: str, db: Session = Depends(get_db)):
    run = db.query(RunRow).filter(RunRow.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    events = (
        db.query(TraceEventRow)
        .filter(TraceEventRow.run_id == run_id)
        .order_by(TraceEventRow.timestamp)
        .all()
    )
    return [_trace_row_to_response(ev) for ev in events]


@router.get(
    "/runs/{run_id}/attribution",
    summary="Counterfactual fault attribution for a run",
)
async def get_run_attribution(run_id: str, db: Session = Depends(get_db)):
    """
    Run counterfactual oracle-replacement attribution on a completed run.

    Returns Shapley values and REF scores for each pipeline component:
    - scope (R): which records were retrieved
    - facts (E): which field values were extracted
    - aggregation (A): how facts were reduced to an answer
    """
    run_row = db.query(RunRow).filter(RunRow.run_id == run_id).first()
    if not run_row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    if run_row.status != "completed":
        raise HTTPException(status_code=422, detail="Attribution requires a completed run")
    
    if not run_row.gold_answer_value:
        raise HTTPException(status_code=422, detail="Attribution requires a gold answer")
    
    q_row = db.query(QueryRow).filter(QueryRow.query_id == run_row.query_id).first()
    if not q_row:
        raise HTTPException(status_code=404, detail="Query not found")
    
    from faulttrace_core.models import GoldAnswer, QuerySpec
    from faulttrace_pipelines.attribution import CounterfactualAttributor
    
    settings = get_settings()
    query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
    gold = GoldAnswer.model_validate(json.loads(q_row.gold_json))
    
    world_dir = settings.data_root / "generated" / "worlds" / query_spec.world_id
    parquet_path = world_dir / "records.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="World data not found for attribution")
    
    df = pd.read_parquet(parquet_path)
    
    # Parse pipeline answer
    raw = run_row.answer
    try:
        import ast
        pipeline_answer = ast.literal_eval(raw)
    except Exception:
        pipeline_answer = raw
    
    attributor = CounterfactualAttributor()
    result = attributor.attribute(
        run_id=run_id,
        query=query_spec,
        pipeline_id=run_row.pipeline_id,
        pipeline_answer=pipeline_answer,
        gold_answer_obj=gold,
        oracle_df=df,
    )
    
    return result.to_dict()


@router.post(
    "/runs/batch-attribution",
    summary="Batch counterfactual fault attribution",
)
async def batch_attribution(
    run_ids: list[str],
    db: Session = Depends(get_db)
):
    """
    Run counterfactual oracle-replacement attribution on a batch of completed runs.
    """
    if not run_ids:
        raise HTTPException(status_code=422, detail="Provide a list of run_ids")
        
    runs_to_process = []
    
    from faulttrace_core.models import GoldAnswer, QuerySpec
    from faulttrace_core.models import PipelineRun
    from faulttrace_pipelines.batch_attribution import BatchAttributionRunner
    import ast
    
    settings = get_settings()
    
    for run_id in run_ids:
        run_row = db.query(RunRow).filter(RunRow.run_id == run_id).first()
        if not run_row or run_row.status != "completed" or not run_row.gold_answer_value:
            continue
            
        q_row = db.query(QueryRow).filter(QueryRow.query_id == run_row.query_id).first()
        if not q_row:
            continue
            
        query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
        gold = GoldAnswer.model_validate(json.loads(q_row.gold_json))
        
        world_dir = settings.data_root / "generated" / "worlds" / query_spec.world_id
        parquet_path = world_dir / "records.parquet"
        if not parquet_path.exists():
            continue
            
        df = pd.read_parquet(parquet_path)
        
        raw = run_row.answer
        try:
            pipeline_answer = ast.literal_eval(raw)
        except Exception:
            pipeline_answer = raw
            
        parent_run = PipelineRun(
            run_id=run_row.run_id,
            query_id=run_row.query_id,
            pipeline_id=run_row.pipeline_id,
            provider_id=run_row.provider_id,
            started_at=run_row.started_at,
            completed_at=run_row.completed_at,
            status=run_row.status,
            answer=pipeline_answer,
            gold_answer_value=run_row.gold_answer_value,
            is_correct=run_row.is_correct,
            is_within_tolerance=run_row.is_correct,
            loss=run_row.loss,
            latency_ms=run_row.latency_ms,
            token_estimate_input=0,
            token_estimate_output=0,
            artifact_references=json.loads(run_row.artifact_refs_json) if run_row.artifact_refs_json else {}
        )
        
        runs_to_process.append((parent_run, query_spec, gold, df))
        
    import uuid
    batch_id = str(uuid.uuid4())
    
    runner = BatchAttributionRunner(artifacts_dir=settings.artifacts_root / "batch_diagnostics")
    summary = runner.run_batch(batch_id, runs_to_process)
    
    return summary


@router.get(
    "/leaderboard",
    summary="Pipeline leaderboard — accuracy and error breakdown",
)
async def get_leaderboard(db: Session = Depends(get_db)):
    """
    Aggregate pipeline runs into a leaderboard with accuracy and mean loss per pipeline.
    """
    from sqlalchemy import func
    
    rows = db.query(
        RunRow.pipeline_id,
        func.count(RunRow.run_id).label("total"),
        func.avg(RunRow.is_correct.cast(type_=type(True))).label("accuracy"),
        func.avg(RunRow.loss).label("mean_loss"),
        func.avg(RunRow.latency_ms).label("mean_latency_ms"),
    ).group_by(RunRow.pipeline_id).all()
    
    board = []
    for r in rows:
        total = r.total or 0
        acc = float(r.accuracy or 0)
        board.append({
            "pipeline_id": r.pipeline_id,
            "total_runs": total,
            "correct_runs": round(acc * total),
            "accuracy": round(acc, 4),
            "mean_loss": round(float(r.mean_loss or 0), 4),
            "mean_latency_ms": round(float(r.mean_latency_ms or 0), 2),
        })
    
    board.sort(key=lambda x: x["accuracy"], reverse=True)
    return {"leaderboard": board, "total_pipelines": len(board)}


@router.get("/runs/{run_id}/certificate", summary="Get certification result for a run")
async def get_run_certificate(run_id: str, db: Session = Depends(get_db)):
    row = db.query(RunRow).filter(RunRow.run_id == run_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
        
    return {
        "certificate_id": row.certificate_id,
        "certificate_hash": row.certificate_hash,
        "policy_decision": row.policy_decision,
        "abstention_reason": row.abstention_reason,
        "final_presented_answer": row.final_presented_answer,
        "raw_answer": row.raw_answer
    }


@router.get("/runs/batch-evaluate-policy", summary="Evaluate a policy across all completed runs")
async def batch_evaluate_policy(policy_id: str = "strict_exact_v1", db: Session = Depends(get_db)):
    from faulttrace_core.models import PipelineRun, CoverageDecision
    from faulttrace_pipelines.selective_metrics import evaluate_certification_policy
    
    # We parse out the basic info required for evaluate_certification_policy
    rows = db.query(RunRow).filter(RunRow.status == "completed").all()
    
    # Simple conversion
    runs = []
    for r in rows:
        runs.append(PipelineRun(
            run_id=r.run_id,
            query_id=r.query_id,
            pipeline_id=r.pipeline_id,
            provider_id=r.provider_id,
            status=r.status,
            loss=r.loss,
            is_correct=r.is_correct,
            policy_decision=r.policy_decision,
            answer=r.answer,
            raw_answer=r.raw_answer
        ))
        
    result = evaluate_certification_policy(runs)
    return result.model_dump()


def _run_row_to_response(row: RunRow) -> RunResponse:
    artifact_refs = json.loads(row.artifact_refs_json) if row.artifact_refs_json else {}
    return RunResponse(
        run_id=row.run_id,
        query_id=row.query_id,
        pipeline_id=row.pipeline_id,
        provider_id=row.provider_id,
        status=row.status,
        answer=row.answer,
        gold_answer_value=row.gold_answer_value,
        is_correct=row.is_correct,
        loss=row.loss,
        latency_ms=row.latency_ms,
        error_message=row.error_message,
        config_hash=row.config_hash,
        artifact_refs=artifact_refs,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


def _trace_row_to_response(row: TraceEventRow) -> TraceEventResponse:
    payload = json.loads(row.payload_json) if row.payload_json else {}
    return TraceEventResponse(
        event_id=row.event_id,
        run_id=row.run_id,
        parent_event_id=row.parent_event_id,
        stage=row.stage,
        event_type=row.event_type,
        message=row.message,
        record_count_in=row.record_count_in,
        record_count_out=row.record_count_out,
        duration_ms=row.duration_ms,
        payload=payload,
        timestamp=row.timestamp,
    )
