"""Queries endpoints."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import get_db, QueryRow
from faulttrace_api.models import GenerateQueriesRequest, PaginatedResponse, QueryResponse

router = APIRouter()


@router.get("/queries", response_model=PaginatedResponse[QueryResponse], summary="List queries with filters")
async def list_queries(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    world_id: Optional[str] = Query(None),
    family: Optional[str] = Query(None),
    template_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(QueryRow)
    if world_id:
        q = q.filter(QueryRow.world_id == world_id)
    if family:
        q = q.filter(QueryRow.family == family)
    if template_id:
        q = q.filter(QueryRow.template_id == template_id)
    
    total = q.count()
    rows = q.offset((page - 1) * page_size).limit(page_size).all()
    
    return PaginatedResponse(
        items=[_query_row_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )


@router.get("/queries/{query_id}", response_model=QueryResponse, summary="Get query details")
async def get_query(query_id: str, db: Session = Depends(get_db)):
    row = db.query(QueryRow).filter(QueryRow.query_id == query_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found")
    return _query_row_to_response(row)


@router.post("/queries/generate", summary="Generate queries for a world")
async def generate_queries(
    request: GenerateQueriesRequest,
    db: Session = Depends(get_db),
):
    """Generate procedural queries for a corpus world and store them."""
    settings = get_settings()
    
    # Check world exists
    from faulttrace_api.database import WorldRow
    world_row = db.query(WorldRow).filter(WorldRow.world_id == request.world_id).first()
    if not world_row:
        raise HTTPException(status_code=404, detail=f"World '{request.world_id}' not found")
    
    import pandas as pd
    from faulttrace_pipelines.query_factory import QueryFactory
    from faulttrace_gold.validator import GoldValidator
    
    world_dir = settings.data_root / "generated" / "worlds" / request.world_id
    parquet_path = world_dir / "records.parquet"
    df = pd.read_parquet(parquet_path)
    
    factory = QueryFactory(data_dir=settings.data_root / "generated")
    validator = GoldValidator()
    queries = factory.generate_for_world(
        world_id=request.world_id,
        target_count=request.count,
        seed=request.seed,
    )
    
    stored = 0
    for q in queries:
        gold_result = validator.validate(q, df, parquet_path)
        gold_json = None
        if gold_result.gold_answer:
            gold_json = json.dumps(gold_result.gold_answer.model_dump(mode="json"), default=str)
        
        existing = db.query(QueryRow).filter(QueryRow.query_id == q.query_id).first()
        if existing:
            continue  # Skip duplicates
        
        row = QueryRow(
            query_id=q.query_id,
            world_id=q.world_id,
            family=q.family.value,
            natural_language_question=q.natural_language_question,
            template_id=q.template_id,
            version=q.version,
            spec_json=json.dumps(q.model_dump(mode="json"), default=str),
            gold_json=gold_json,
            created_at=q.created_at.replace(tzinfo=None),
        )
        db.add(row)
        stored += 1
    
    db.commit()
    return {"generated": len(queries), "stored": stored, "world_id": request.world_id}


def _query_row_to_response(row: QueryRow) -> QueryResponse:
    spec = json.loads(row.spec_json) if row.spec_json else {}
    gold = json.loads(row.gold_json) if row.gold_json else None
    return QueryResponse(
        query_id=row.query_id,
        world_id=row.world_id,
        family=row.family,
        natural_language_question=row.natural_language_question,
        template_id=row.template_id,
        version=row.version,
        spec=spec,
        gold=gold,
        created_at=row.created_at,
    )
