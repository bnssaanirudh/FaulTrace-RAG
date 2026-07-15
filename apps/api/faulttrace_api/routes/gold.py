"""Gold answer endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from faulttrace_api.database import get_db, QueryRow

router = APIRouter()


@router.get("/gold/{query_id}", summary="Get gold answer for a query")
async def get_gold_answer(query_id: str, db: Session = Depends(get_db)):
    row = db.query(QueryRow).filter(QueryRow.query_id == query_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found")
    if not row.gold_json:
        raise HTTPException(
            status_code=404,
            detail=f"No gold answer computed for query '{query_id}'",
        )
    return json.loads(row.gold_json)
