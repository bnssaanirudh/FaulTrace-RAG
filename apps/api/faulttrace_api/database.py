"""
Database setup: SQLAlchemy 2 engine, session factory, and ORM models.

App metadata (worlds, queries, runs, traces) is stored in SQLite by default.
Set FAULTTRACE_DATABASE_URL to upgrade to PostgreSQL.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from faulttrace_api.config import get_settings

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class WorldRow(Base):
    __tablename__ = "worlds"

    world_id = Column(String, primary_key=True)
    dataset_id = Column(String, default="track_m")
    seed = Column(Integer, nullable=False)
    scale_n = Column(Integer, nullable=False)
    parent_world_id = Column(String, nullable=True)
    creation_policy = Column(String, default="deterministic_superset")
    record_ids_hash = Column(String, nullable=False)
    manifest_path = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    schema_version = Column(String, default="1.0.0")


class QueryRow(Base):
    __tablename__ = "queries"

    query_id = Column(String, primary_key=True)
    world_id = Column(String, nullable=False)
    family = Column(String, nullable=False)
    natural_language_question = Column(Text, nullable=False)
    template_id = Column(String, default="manual")
    version = Column(String, default="1.0")
    spec_json = Column(Text, nullable=False)  # Full QuerySpec JSON
    gold_json = Column(Text, nullable=True)   # GoldAnswer JSON
    created_at = Column(DateTime, nullable=False)


class RunRow(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    query_id = Column(String, nullable=False)
    pipeline_id = Column(String, nullable=False)
    provider_id = Column(String, default="deterministic")
    status = Column(String, default="pending")
    answer = Column(Text, nullable=True)
    gold_answer_value = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    loss = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    config_hash = Column(String, nullable=True)
    artifact_refs_json = Column(Text, default="{}")
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Certification (Prompt 6)
    raw_answer = Column(Text, nullable=True)
    policy_decision = Column(String, nullable=True)
    final_presented_answer = Column(Text, nullable=True)
    abstention_reason = Column(Text, nullable=True)
    certificate_id = Column(String, nullable=True)
    certificate_hash = Column(String, nullable=True)

    schema_version = Column(String, default="1.0.0")


class TraceEventRow(Base):
    __tablename__ = "trace_events"

    event_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    parent_event_id = Column(String, nullable=True)
    stage = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    message = Column(Text, default="")
    record_count_in = Column(Integer, nullable=True)
    record_count_out = Column(Integer, nullable=True)
    duration_ms = Column(Float, nullable=True)
    payload_json = Column(Text, default="{}")
    timestamp = Column(DateTime, nullable=False)
    schema_version = Column(String, default="1.0.0")


# ---------------------------------------------------------------------------
# Engine and session
# ---------------------------------------------------------------------------


def get_engine(db_url: Optional[str] = None):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if db_url:
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        return engine
    if _engine is None:
        settings = get_settings()
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            settings.effective_database_url,
            connect_args={"check_same_thread": False},
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db():
    """FastAPI dependency: yield a database session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
