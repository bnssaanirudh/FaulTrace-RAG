"""Worlds endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import get_db, WorldRow
from faulttrace_api.models import PaginatedResponse, RecordResponse, WorldResponse

router = APIRouter()


@router.get("/worlds", response_model=PaginatedResponse[WorldResponse], summary="List all corpus worlds")
async def list_worlds(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    total = db.query(WorldRow).count()
    start = (page - 1) * page_size
    rows = db.query(WorldRow).order_by(WorldRow.scale_n).offset(start).limit(page_size).all()
    
    return PaginatedResponse(
        items=[_world_row_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(start + page_size) < total,
    )


@router.get("/worlds/{world_id}", response_model=WorldResponse, summary="Get world details")
async def get_world(world_id: str, db: Session = Depends(get_db)):
    row = db.query(WorldRow).filter(WorldRow.world_id == world_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")
    return _world_row_to_response(row)


@router.get(
    "/worlds/{world_id}/records",
    response_model=PaginatedResponse,
    summary="List records in a world with pagination and filters",
)
async def list_world_records(
    world_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_rating: Optional[float] = Query(None, ge=1.0, le=5.0),
    max_rating: Optional[float] = Query(None, ge=1.0, le=5.0),
    verified_only: bool = Query(False),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    world_dir = settings.data_root / "generated" / "worlds" / world_id
    parquet_path = world_dir / "records.parquet"
    
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"World data not found for '{world_id}'")
    
    df = pd.read_parquet(parquet_path)
    
    # Apply filters
    if category:
        df = df[df["category"] == category]
    if brand:
        df = df[df["brand"] == brand]
    if min_rating is not None:
        df = df[df["rating"] >= min_rating]
    if max_rating is not None:
        df = df[df["rating"] <= max_rating]
    if verified_only:
        df = df[df["verified_purchase"] == True]
    
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]
    
    items = []
    for _, row in page_df.iterrows():
        items.append(RecordResponse(
            record_id=row["record_id"],
            product_id=row["product_id"],
            category=row["category"],
            title=row["title"],
            brand=row["brand"],
            rating=float(row["rating"]),
            helpful_votes=int(row["helpful_votes"]),
            verified_purchase=bool(row["verified_purchase"]),
            event_time=row["event_time"],
            price=float(row["price"]) if row["price"] is not None and not pd.isna(row["price"]) else None,
            text=str(row.get("text", "")),
            world_id=row["world_id"],
        ))
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
    )


def _world_row_to_response(row: WorldRow) -> WorldResponse:
    return WorldResponse(
        world_id=row.world_id,
        dataset_id=row.dataset_id,
        seed=row.seed,
        scale_n=row.scale_n,
        parent_world_id=row.parent_world_id,
        creation_policy=row.creation_policy,
        record_ids_hash=row.record_ids_hash,
        manifest_path=row.manifest_path,
        created_at=row.created_at,
        schema_version=row.schema_version,
    )
