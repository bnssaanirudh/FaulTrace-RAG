"""Shared Pydantic response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool


class WorldResponse(BaseModel):
    world_id: str
    dataset_id: str
    seed: int
    scale_n: int
    parent_world_id: Optional[str] = None
    creation_policy: str
    record_ids_hash: str
    manifest_path: str
    created_at: datetime
    schema_version: str


class QueryResponse(BaseModel):
    query_id: str
    world_id: str
    family: str
    natural_language_question: str
    template_id: str
    version: str
    spec: dict[str, Any]
    gold: Optional[dict[str, Any]] = None
    created_at: datetime


class RunResponse(BaseModel):
    run_id: str
    query_id: str
    pipeline_id: str
    provider_id: str
    status: str
    answer: Optional[str] = None
    gold_answer_value: Optional[str] = None
    is_correct: Optional[bool] = None
    loss: Optional[float] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    config_hash: Optional[str] = None
    artifact_refs: dict[str, str] = {}
    started_at: datetime
    completed_at: Optional[datetime] = None


class TraceEventResponse(BaseModel):
    event_id: str
    run_id: str
    parent_event_id: Optional[str] = None
    stage: str
    event_type: str
    message: str
    record_count_in: Optional[int] = None
    record_count_out: Optional[int] = None
    duration_ms: Optional[float] = None
    payload: dict[str, Any] = {}
    timestamp: datetime


class RecordResponse(BaseModel):
    record_id: str
    product_id: str
    category: str
    title: str
    brand: str
    rating: float
    helpful_votes: int
    verified_purchase: bool
    event_time: datetime
    price: Optional[float] = None
    text: str
    world_id: str


class CreateRunRequest(BaseModel):
    query_id: str
    pipeline_id: str = "P0-deterministic-scope-baseline"
    provider_id: str = "deterministic"


class GenerateQueriesRequest(BaseModel):
    world_id: str
    count: int = 60
    seed: Optional[int] = None


class SeedDemoRequest(BaseModel):
    seed: int = 42
    scales: list[int] = [10, 50, 200, 1000]
    overwrite: bool = False
