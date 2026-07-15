"""
Core contract unit tests for FaultTrace-RAG.

Tests: validation, stable hashing, serialization round trips, invalid predicates.
"""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from faulttrace_core.models import (
    AndPredicate,
    AggregationSpec,
    ComparisonSpec,
    CorpusRecord,
    CorpusWorld,
    CountSpec,
    EqPredicate,
    FactSpec,
    GoldAnswer,
    GoldAnswer,
    InPredicate,
    IsNotNullPredicate,
    IsNullPredicate,
    MeanSpec,
    NeqPredicate,
    NullPolicy,
    OrPredicate,
    PipelineRun,
    ProportionSpec,
    QueryFamily,
    QuerySpec,
    RangePredicate,
    RecordCategory,
    RunStatus,
    TopKSpec,
    TrendSpec,
    TraceEvent,
    TraceEventType,
    AgreementStatus,
)
from faulttrace_core.predicates import PredicateCompiler, _validate_field_name


# ---------------------------------------------------------------------------
# CorpusRecord tests
# ---------------------------------------------------------------------------

def make_record(**kwargs) -> CorpusRecord:
    defaults = dict(
        record_id="rec_0001_000000",
        source_record_id="R0001_000000",
        world_id="world_s42_n10",
        product_id="BABC123456",
        category=RecordCategory.ELECTRONICS,
        title="Test Product",
        brand="TechPrime",
        rating=4.5,
        helpful_votes=10,
        verified_purchase=True,
        event_time=datetime(2022, 3, 15, tzinfo=timezone.utc),
        raw_payload_hash="a" * 32,
    )
    defaults.update(kwargs)
    return CorpusRecord(**defaults)


def test_corpus_record_valid():
    r = make_record()
    assert r.rating == 4.5
    assert r.category == RecordCategory.ELECTRONICS


def test_corpus_record_rating_bounds():
    with pytest.raises(Exception):
        make_record(rating=0.5)
    with pytest.raises(Exception):
        make_record(rating=5.5)


def test_corpus_record_rating_rounding():
    r = make_record(rating=4.0)
    assert r.rating == 4.0


def test_corpus_record_content_hash_stable():
    r1 = make_record()
    r2 = make_record()
    assert r1.content_hash() == r2.content_hash()


def test_corpus_record_serialization_round_trip():
    r = make_record(price=Decimal("29.99"))
    data = json.loads(r.model_dump_json_bytes())
    assert data["record_id"] == "rec_0001_000000"
    assert data["category"] == "Electronics"


# ---------------------------------------------------------------------------
# ScopePredicate tests
# ---------------------------------------------------------------------------

def test_eq_predicate_valid():
    p = EqPredicate(field="category", value="Electronics")
    assert p.kind == "eq"
    assert p.field == "category"


def test_in_predicate_empty_invalid():
    with pytest.raises(Exception):
        InPredicate(field="category", values=[])


def test_range_predicate_both_none_invalid():
    with pytest.raises(Exception):
        RangePredicate(field="rating", low=None, high=None)


def test_and_predicate_single_operand_invalid():
    with pytest.raises(Exception):
        AndPredicate(operands=[EqPredicate(field="rating", value=5.0)])


def test_or_predicate_valid():
    p = OrPredicate(operands=[
        EqPredicate(field="category", value="Electronics"),
        EqPredicate(field="category", value="Books"),
    ])
    assert len(p.operands) == 2


def test_predicate_serialization():
    p = AndPredicate(operands=[
        EqPredicate(field="category", value="Electronics"),
        RangePredicate(field="rating", low=4.0),
    ])
    data = p.model_dump(mode="json")
    assert data["kind"] == "and"
    assert len(data["operands"]) == 2


# ---------------------------------------------------------------------------
# PredicateCompiler tests
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4", "r5"],
        "category": ["Electronics", "Books", "Electronics", "Books", "Electronics"],
        "rating": [4.5, 3.0, 5.0, 4.0, 2.0],
        "price": [29.99, None, 49.99, 15.00, None],
        "verified_purchase": [True, False, True, True, False],
        "helpful_votes": [10, 0, 5, 2, 0],
        "event_time": pd.to_datetime([
            "2022-01-15", "2021-06-20", "2022-07-01", "2020-11-30", "2023-03-10"
        ], utc=True),
        "brand": ["TechPrime", "BookCo", "TechPrime", "BookCo", "VoltEdge"],
        "world_id": ["w1"] * 5,
    })


def test_compiler_eq_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(EqPredicate(field="category", value="Electronics"), sample_df)
    assert mask.sum() == 3


def test_compiler_neq_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(NeqPredicate(field="category", value="Electronics"), sample_df)
    assert mask.sum() == 2


def test_compiler_in_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(InPredicate(field="category", values=["Electronics"]), sample_df)
    assert mask.sum() == 3


def test_compiler_range_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(RangePredicate(field="rating", low=4.0), sample_df)
    assert mask.sum() == 3  # 4.5, 5.0, 4.0


def test_compiler_is_null_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(IsNullPredicate(field="price"), sample_df)
    assert mask.sum() == 2


def test_compiler_is_not_null_predicate(sample_df):
    compiler = PredicateCompiler()
    mask = compiler.to_pandas_mask(IsNotNullPredicate(field="price"), sample_df)
    assert mask.sum() == 3


def test_compiler_and_predicate(sample_df):
    compiler = PredicateCompiler()
    pred = AndPredicate(operands=[
        EqPredicate(field="category", value="Electronics"),
        EqPredicate(field="verified_purchase", value=True),
    ])
    mask = compiler.to_pandas_mask(pred, sample_df)
    assert mask.sum() == 2  # r1, r3


def test_compiler_or_predicate(sample_df):
    compiler = PredicateCompiler()
    pred = OrPredicate(operands=[
        EqPredicate(field="category", value="Electronics"),
        EqPredicate(field="category", value="Books"),
    ])
    mask = compiler.to_pandas_mask(pred, sample_df)
    assert mask.sum() == 5


def test_compiler_sql_eq(sample_df):
    compiler = PredicateCompiler()
    sql = compiler.to_duckdb_sql(EqPredicate(field="category", value="Electronics"))
    assert "category" in sql
    assert "Electronics" in sql
    assert "eval" not in sql.lower()


def test_compiler_sql_injection_safe():
    """Ensure single quotes in string values are escaped."""
    compiler = PredicateCompiler()
    sql = compiler.to_duckdb_sql(EqPredicate(field="brand", value="O'Reilly"))
    assert "O''Reilly" in sql  # escaped
    assert "eval" not in sql.lower()


def test_compiler_disallows_unknown_field():
    compiler = PredicateCompiler()
    with pytest.raises(ValueError, match="not in allowed fields"):
        compiler.to_pandas_mask(
            EqPredicate(field="__class__", value="x"),
            pd.DataFrame({"__class__": ["x"]})
        )


# ---------------------------------------------------------------------------
# QuerySpec tests
# ---------------------------------------------------------------------------

def test_query_spec_spec_hash_stable():
    q1 = QuerySpec(
        family=QueryFamily.COUNT,
        natural_language_question="How many reviews are there?",
        scope_predicate=EqPredicate(field="category", value="Electronics"),
        fact_spec=FactSpec(fields=["record_id"]),
        aggregation_spec=CountSpec(),
        world_id="world_s42_n10",
    )
    q2 = QuerySpec(
        query_id=q1.query_id,  # same query_id
        family=QueryFamily.COUNT,
        natural_language_question="How many reviews are there?",
        scope_predicate=EqPredicate(field="category", value="Electronics"),
        fact_spec=FactSpec(fields=["record_id"]),
        aggregation_spec=CountSpec(),
        world_id="world_s42_n10",
    )
    assert q1.spec_hash() == q2.spec_hash()


def test_query_spec_question_too_short():
    with pytest.raises(Exception):
        QuerySpec(
            family=QueryFamily.COUNT,
            natural_language_question="?",  # too short
            scope_predicate=EqPredicate(field="category", value="Electronics"),
            fact_spec=FactSpec(fields=["record_id"]),
            aggregation_spec=CountSpec(),
            world_id="world_s42_n10",
        )


# ---------------------------------------------------------------------------
# GoldAnswer tests
# ---------------------------------------------------------------------------

def test_gold_answer_hash():
    g = GoldAnswer(
        query_id="qid1",
        world_id="w1",
        answer_value=42,
        eligible_record_count=100,
        evidence_hash="abc123" + "0" * 26,
        agreement_status=AgreementStatus.AGREED,
    )
    h = g.answer_hash()
    assert len(h) == 32  # SHA-256 truncated to 32 chars


# ---------------------------------------------------------------------------
# PipelineRun tests  
# ---------------------------------------------------------------------------

def test_pipeline_run_config_hash():
    q = QuerySpec(
        family=QueryFamily.COUNT,
        natural_language_question="How many reviews are there?",
        scope_predicate=EqPredicate(field="category", value="Electronics"),
        fact_spec=FactSpec(fields=["record_id"]),
        aggregation_spec=CountSpec(),
        world_id="world_s42_n10",
    )
    run = PipelineRun(
        query_id=q.query_id,
        pipeline_id="P0-deterministic-scope-baseline",
    )
    h1 = run.compute_config_hash(q)
    h2 = run.compute_config_hash(q)
    assert h1 == h2
    assert len(h1) == 32


# ---------------------------------------------------------------------------
# TraceEvent tests
# ---------------------------------------------------------------------------

def test_trace_event_valid():
    ev = TraceEvent(
        run_id="run_001",
        stage="query_load",
        event_type=TraceEventType.QUERY_LOAD,
        message="Query loaded",
    )
    assert ev.stage == "query_load"
    assert ev.event_type == TraceEventType.QUERY_LOAD


# ---------------------------------------------------------------------------
# FactSpec tests
# ---------------------------------------------------------------------------

def test_fact_spec_empty_fields_invalid():
    with pytest.raises(Exception):
        FactSpec(fields=[])


def test_fact_spec_valid():
    fs = FactSpec(fields=["record_id", "rating", "category"])
    assert len(fs.fields) == 3
