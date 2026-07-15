"""
Gold engine tests: validates Pandas and DuckDB evaluators agree on all families.

Tests: count, mean, proportion, comparison, top-k, trend, nulls, ties, boundary dates.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from faulttrace_core.models import (
    AndPredicate,
    ComparisonSpec,
    CountSpec,
    EqPredicate,
    FactSpec,
    IsNotNullPredicate,
    IsNullPredicate,
    MeanSpec,
    NullPolicy,
    ProportionSpec,
    QueryFamily,
    QuerySpec,
    RangePredicate,
    TiePolicy,
    TopKSpec,
    TrendSpec,
)
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_gold.duckdb_engine import DuckDBEvaluator
from faulttrace_gold.validator import GoldValidator, _results_agree


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample corpus DataFrame for testing."""
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(20)],
        "category": (
            ["Electronics"] * 8 + ["Books"] * 7 + ["Sports"] * 5
        ),
        "brand": (
            ["TechPrime"] * 5 + ["VoltEdge"] * 3 +
            ["BookCo"] * 4 + ["LitGuild"] * 3 +
            ["SportPeak"] * 3 + ["FitCore"] * 2
        ),
        "rating": [
            4.5, 5.0, 3.0, 4.0, 2.0, 4.5, 5.0, 3.5,  # Electronics
            4.0, 3.0, 5.0, 4.5, 2.0, 3.5, 4.0,        # Books
            3.0, 4.0, 5.0, 3.5, 4.0,                   # Sports
        ],
        "price": [
            29.99, 49.99, None, 19.99, 9.99, 39.99, 59.99, None,
            14.99, 9.99, 24.99, None, 7.99, 12.99, 19.99,
            49.99, 29.99, 79.99, None, 39.99,
        ],
        "verified_purchase": [
            True, True, False, True, False, True, True, False,
            True, False, True, True, False, True, True,
            True, True, False, True, True,
        ],
        "helpful_votes": [
            10, 0, 5, 2, 0, 8, 15, 1,
            3, 0, 7, 4, 0, 2, 5,
            0, 6, 12, 0, 3,
        ],
        "event_time": pd.to_datetime([
            "2022-01-15", "2022-03-20", "2021-11-05", "2022-06-01",
            "2020-08-15", "2022-12-31", "2023-01-01", "2021-06-20",
            "2022-04-10", "2021-09-15", "2022-07-20", "2020-12-01",
            "2023-02-14", "2022-11-30", "2021-03-05",
            "2022-05-20", "2021-12-10", "2022-08-08", "2020-10-25", "2023-03-15",
        ], utc=True),
        "world_id": ["test_world"] * 20,
    })


@pytest.fixture
def parquet_path(sample_df, tmp_path) -> Path:
    """Save sample_df to parquet and return path."""
    path = tmp_path / "records.parquet"
    sample_df.to_parquet(path, index=False)
    return path


def make_query(family, agg_spec, scope_pred, fields=None):
    if fields is None:
        fields = ["record_id", "category", "rating"]
    return QuerySpec(
        family=family,
        natural_language_question="Test question for gold engine testing",
        scope_predicate=scope_pred,
        fact_spec=FactSpec(fields=fields),
        aggregation_spec=agg_spec,
        world_id="test_world",
        template_id="test",
        tolerance=1e-4,
    )


# ---------------------------------------------------------------------------
# Count tests
# ---------------------------------------------------------------------------

def test_count_all(sample_df, parquet_path):
    query = make_query(
        QueryFamily.COUNT,
        CountSpec(),
        RangePredicate(field="rating", low=1.0, high=5.0),
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value == 20


def test_count_category_filter(sample_df, parquet_path):
    query = make_query(
        QueryFamily.COUNT,
        CountSpec(),
        EqPredicate(field="category", value="Electronics"),
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value == 8


def test_count_verified_filter(sample_df, parquet_path):
    query = make_query(
        QueryFamily.COUNT,
        CountSpec(),
        EqPredicate(field="verified_purchase", value=True),
        fields=["record_id", "verified_purchase"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    verified_count = sample_df["verified_purchase"].sum()
    assert result.gold_answer.answer_value == int(verified_count)


def test_count_empty_scope(sample_df, parquet_path):
    """Empty scope should return 0."""
    query = make_query(
        QueryFamily.COUNT,
        CountSpec(),
        EqPredicate(field="rating", value=6.0),  # impossible
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value == 0


# ---------------------------------------------------------------------------
# Mean tests
# ---------------------------------------------------------------------------

def test_mean_rating_electronics(sample_df, parquet_path):
    query = make_query(
        QueryFamily.MEAN,
        MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
        EqPredicate(field="category", value="Electronics"),
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    
    # Verify manually
    elec = sample_df[sample_df["category"] == "Electronics"]["rating"]
    expected = round(float(elec.mean()), 4)
    assert abs(float(result.gold_answer.answer_value) - expected) < 1e-3


def test_mean_price_excludes_nulls(sample_df, parquet_path):
    """Mean price should exclude null values."""
    query = make_query(
        QueryFamily.MEAN,
        MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
        IsNotNullPredicate(field="price"),
        fields=["record_id", "price"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value is not None


def test_mean_empty_scope(sample_df, parquet_path):
    """Mean on empty scope should return None."""
    query = make_query(
        QueryFamily.MEAN,
        MeanSpec(field="rating"),
        EqPredicate(field="rating", value=6.0),  # empty
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value is None


# ---------------------------------------------------------------------------
# Proportion tests
# ---------------------------------------------------------------------------

def test_proportion_verified_electronics(sample_df, parquet_path):
    query = make_query(
        QueryFamily.PROPORTION,
        ProportionSpec(
            numerator_predicate=EqPredicate(field="verified_purchase", value=True)
        ),
        EqPredicate(field="category", value="Electronics"),
        fields=["record_id", "category", "verified_purchase"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    
    # Manually: Electronics verified / Electronics total
    elec = sample_df[sample_df["category"] == "Electronics"]
    expected = round(elec["verified_purchase"].mean(), 4)
    assert abs(float(result.gold_answer.answer_value) - expected) < 1e-3


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

def test_comparison_mean_rating(sample_df, parquet_path):
    query = make_query(
        QueryFamily.COMPARISON,
        ComparisonSpec(
            measure="mean",
            field="rating",
            group_a_predicate=EqPredicate(field="category", value="Electronics"),
            group_b_predicate=EqPredicate(field="category", value="Books"),
            output="difference",
        ),
        RangePredicate(field="rating", low=1.0),
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed


def test_comparison_count(sample_df, parquet_path):
    query = make_query(
        QueryFamily.COMPARISON,
        ComparisonSpec(
            measure="count",
            group_a_predicate=EqPredicate(field="category", value="Electronics"),
            group_b_predicate=EqPredicate(field="category", value="Sports"),
            output="difference",
        ),
        RangePredicate(field="rating", low=1.0),
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value == 3.0  # Electronics(8) - Sports(5)


# ---------------------------------------------------------------------------
# Top-K tests
# ---------------------------------------------------------------------------

def test_topk_brands_by_count(sample_df, parquet_path):
    query = make_query(
        QueryFamily.TOP_K,
        TopKSpec(group_by_field="brand", measure="count", k=3, tie_policy=TiePolicy.FIRST),
        RangePredicate(field="rating", low=1.0),
        fields=["record_id", "brand"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert isinstance(result.gold_answer.answer_value, list)
    assert len(result.gold_answer.answer_value) == 3


def test_topk_categories_by_mean_rating(sample_df, parquet_path):
    query = make_query(
        QueryFamily.TOP_K,
        TopKSpec(
            group_by_field="category", measure="mean", value_field="rating",
            k=2, tie_policy=TiePolicy.FIRST
        ),
        RangePredicate(field="rating", low=1.0),
        fields=["record_id", "category", "rating"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert len(result.gold_answer.answer_value) == 2


# ---------------------------------------------------------------------------
# Trend tests
# ---------------------------------------------------------------------------

def test_trend_monthly_count(sample_df, parquet_path):
    query = make_query(
        QueryFamily.TREND,
        TrendSpec(bucket="month", measure="count"),
        RangePredicate(field="rating", low=1.0),
        fields=["record_id", "event_time"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed
    assert isinstance(result.gold_answer.answer_value, list)
    assert len(result.gold_answer.answer_value) > 0


def test_trend_quarterly_mean_rating(sample_df, parquet_path):
    query = make_query(
        QueryFamily.TREND,
        TrendSpec(bucket="quarter", measure="mean", value_field="rating"),
        RangePredicate(field="rating", low=1.0),
        fields=["record_id", "event_time", "rating"],
    )
    validator = GoldValidator()
    result = validator.validate(query, sample_df, parquet_path)
    assert result.agreed


# ---------------------------------------------------------------------------
# Agreement tests
# ---------------------------------------------------------------------------

def test_results_agree_floats():
    assert _results_agree(1.0, 1.0, 1e-6)
    assert _results_agree(1.0, 1.0 + 1e-7, 1e-6)
    assert not _results_agree(1.0, 1.1, 1e-6)


def test_results_agree_none():
    assert _results_agree(None, None, 1e-6)
    assert not _results_agree(None, 1.0, 1e-6)


def test_results_agree_lists():
    a = [{"brand": "A", "value": 10.0}, {"brand": "B", "value": 8.0}]
    b = [{"brand": "A", "value": 10.0}, {"brand": "B", "value": 8.0}]
    assert _results_agree(a, b, 1e-6)


def test_disagreement_detection(sample_df, parquet_path):
    """Injecting a disagreement: manually test that it's caught."""
    from faulttrace_gold.validator import _results_agree
    assert not _results_agree(42.0, 43.0, 0.0)
    assert _results_agree(42.0, 42.0 + 1e-8, 1e-6)


# ---------------------------------------------------------------------------
# Null policy tests
# ---------------------------------------------------------------------------

def test_null_policy_exclude_mean(sample_df, parquet_path):
    """EXCLUDE null policy: nulls should not affect mean."""
    query = make_query(
        QueryFamily.MEAN,
        MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE),
        RangePredicate(field="rating", low=1.0),
        fields=["record_id", "price"],
    )
    pd_eval = PandasEvaluator()
    dk_eval = DuckDBEvaluator()
    
    pd_result = pd_eval.evaluate(query, sample_df)
    dk_result = dk_eval.evaluate_from_df(query, sample_df)
    
    assert pd_result["result"] is not None
    assert abs(float(pd_result["result"]) - float(dk_result["result"])) < 1e-3


# ---------------------------------------------------------------------------
# Boundary date tests
# ---------------------------------------------------------------------------

def test_boundary_dates(tmp_path):
    """Records at exact date boundaries should be handled correctly."""
    boundary_df = pd.DataFrame({
        "record_id": ["b1", "b2", "b3"],
        "category": ["Electronics"] * 3,
        "rating": [4.0, 4.5, 5.0],
        "brand": ["TechPrime"] * 3,
        "verified_purchase": [True] * 3,
        "helpful_votes": [0] * 3,
        "price": [29.99] * 3,
        "event_time": pd.to_datetime([
            "2021-12-31T23:59:59+00:00",
            "2022-01-01T00:00:00+00:00",
            "2022-01-01T00:00:01+00:00",
        ]),
        "world_id": ["boundary_world"] * 3,
    })
    
    parquet_path = tmp_path / "boundary.parquet"
    boundary_df.to_parquet(parquet_path, index=False)
    
    cutoff = "2022-01-01T00:00:00+00:00"
    query = make_query(
        QueryFamily.COUNT,
        CountSpec(),
        RangePredicate(field="event_time", low=cutoff, low_inclusive=True),
        fields=["record_id", "event_time"],
    )
    query = query.model_copy(update={"world_id": "boundary_world"})
    
    validator = GoldValidator()
    result = validator.validate(query, boundary_df, parquet_path)
    assert result.agreed
    assert result.gold_answer.answer_value == 2  # b2, b3
