"""Shared Pydantic response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None


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
    parent_world_id: str | None = None
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
    gold: dict[str, Any] | None = None
    created_at: datetime


class RunResponse(BaseModel):
    run_id: str
    query_id: str
    pipeline_id: str
    provider_id: str
    status: str
    answer: str | None = None
    gold_answer_value: str | None = None
    is_correct: bool | None = None
    loss: float | None = None
    latency_ms: float | None = None
    error_message: str | None = None
    config_hash: str | None = None
    artifact_refs: dict[str, str] = {}
    started_at: datetime
    completed_at: datetime | None = None


class TraceEventResponse(BaseModel):
    event_id: str
    run_id: str
    parent_event_id: str | None = None
    stage: str
    event_type: str
    message: str
    record_count_in: int | None = None
    record_count_out: int | None = None
    duration_ms: float | None = None
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
    price: float | None = None
    text: str
    world_id: str


class CreateRunRequest(BaseModel):
    query_id: str
    pipeline_id: str = "P0-deterministic-scope-baseline"
    provider_id: str = "deterministic"


class GenerateQueriesRequest(BaseModel):
    world_id: str
    count: int = 60
    seed: int | None = None


class SeedDemoRequest(BaseModel):
    seed: int = 42
    scales: list[int] = [10, 50, 200, 1000]
    overwrite: bool = False
