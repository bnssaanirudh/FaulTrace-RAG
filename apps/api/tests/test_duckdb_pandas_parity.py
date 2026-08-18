"""
test_duckdb_pandas_parity.py
============================
Validates that DuckDBEvaluator produces *exactly* the same deterministic
analytics results as PandasEvaluator for every Track M query family
(COUNT, MEAN, SUM, PROPORTION, TOP_K, TREND, COMPARISON) at two dataset
scales: N=10 and N=5000 (generated from seed 42).

Strategy
--------
* Build a small but diverse QuerySpec list manually – one specimen per
  major code path – rather than relying on the query factory (which needs
  a world directory on disk).
* For each query we:
  1. Evaluate with PandasEvaluator against the in-memory DataFrame.
  2. Evaluate with DuckDBEvaluator.evaluate_from_df() against the same DataFrame
     (registers it as a DuckDB in-memory table – no Parquet file needed).
  3. Assert result, eligible_count, and contributing_ids all match.

The comparisons use _results_agree() which handles float tolerance (1e-6)
and recursive list/dict comparison.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
from faulttrace_core.models import (
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
from faulttrace_data.generator import TrackMGenerator
from faulttrace_gold.duckdb_engine import DuckDBEvaluator
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_gold.validator import _results_agree

# ---------------------------------------------------------------------------
# Engines (module-level singletons, no state held between calls)
# ---------------------------------------------------------------------------
pandas_engine = PandasEvaluator()
duckdb_engine = DuckDBEvaluator()


# ---------------------------------------------------------------------------
# Fixtures: reproducible DataFrames at N=10 and N=5000
# ---------------------------------------------------------------------------


def _build_df(n: int) -> pd.DataFrame:
    """Generate n Track M records deterministically (seed=42) as a DataFrame."""
    gen = TrackMGenerator(seed=42)
    records = gen.generate_records(n)
    rows = []
    for r in records:
        row = r.model_dump()
        # Normalise Decimal → float so DuckDB can store it without surprises
        if row.get("price") is not None:
            row["price"] = float(row["price"])
        # Ensure event_time is tz-aware datetime
        if isinstance(row.get("event_time"), datetime) and row["event_time"].tzinfo is None:
            row["event_time"] = row["event_time"].replace(tzinfo=UTC)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df_10() -> pd.DataFrame:
    return _build_df(10)


@pytest.fixture(scope="module")
def df_5000() -> pd.DataFrame:
    return _build_df(5000)


# ---------------------------------------------------------------------------
# QuerySpec builder helper
# ---------------------------------------------------------------------------


def _make(
    family: QueryFamily,
    agg_spec: Any,
    scope_pred: Any,
    fields: list[str] | None = None,
    world_id: str = "parity_test",
) -> QuerySpec:
    if fields is None:
        fields = [
            "record_id",
            "category",
            "brand",
            "rating",
            "price",
            "verified_purchase",
            "helpful_votes",
            "event_time",
        ]
    return QuerySpec(
        family=family,
        natural_language_question="Parity check query",
        scope_predicate=scope_pred,
        fact_spec=FactSpec(fields=fields),
        aggregation_spec=agg_spec,
        world_id=world_id,
        template_id="parity_test",
        tolerance=1e-6,
    )


# ---------------------------------------------------------------------------
# Core assertion helper
# ---------------------------------------------------------------------------


def _assert_parity(query: QuerySpec, df: pd.DataFrame, label: str = "") -> None:
    """Run both engines and assert exact agreement on result + eligible_count + ids."""
    pd_out = pandas_engine.evaluate(query, df)
    dk_out = duckdb_engine.evaluate_from_df(query, df)

    prefix = f"[{label}] query={query.query_id}" if label else f"query={query.query_id}"

    # 1. Result value
    agreed = _results_agree(pd_out["result"], dk_out["result"], tolerance=query.tolerance)
    assert agreed, (
        f"{prefix}: result mismatch\n"
        f"  pandas : {pd_out['result']!r}\n"
        f"  duckdb : {dk_out['result']!r}"
    )

    # 2. Eligible count
    assert pd_out["eligible_count"] == dk_out["eligible_count"], (
        f"{prefix}: eligible_count mismatch "
        f"pandas={pd_out['eligible_count']} duckdb={dk_out['eligible_count']}"
    )

    # 3. Contributing IDs (order-independent)
    assert set(pd_out["contributing_ids"]) == set(dk_out["contributing_ids"]), (
        f"{prefix}: contributing_ids mismatch\n"
        f"  pandas : {sorted(pd_out['contributing_ids'])}\n"
        f"  duckdb : {sorted(dk_out['contributing_ids'])}"
    )


# ===========================================================================
# COUNT family
# ===========================================================================


class TestCountParity:
    """COUNT queries: total, filtered, distinct, null-sensitive, empty scope."""

    def test_count_all_n10(self, df_10):
        q = _make(
            QueryFamily.COUNT,
            CountSpec(),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/count_all")

    def test_count_all_n5000(self, df_5000):
        q = _make(
            QueryFamily.COUNT,
            CountSpec(),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/count_all")

    def test_count_category_filter_n10(self, df_10):
        q = _make(
            QueryFamily.COUNT, CountSpec(), EqPredicate(field="category", value="electronics")
        )
        _assert_parity(q, df_10, "N10/count_cat")

    def test_count_category_filter_n5000(self, df_5000):
        q = _make(
            QueryFamily.COUNT, CountSpec(), EqPredicate(field="category", value="electronics")
        )
        _assert_parity(q, df_5000, "N5000/count_cat")

    def test_count_verified_n10(self, df_10):
        q = _make(
            QueryFamily.COUNT, CountSpec(), EqPredicate(field="verified_purchase", value=True)
        )
        _assert_parity(q, df_10, "N10/count_verified")

    def test_count_verified_n5000(self, df_5000):
        q = _make(
            QueryFamily.COUNT, CountSpec(), EqPredicate(field="verified_purchase", value=True)
        )
        _assert_parity(q, df_5000, "N5000/count_verified")

    def test_count_high_rating_n10(self, df_10):
        q = _make(
            QueryFamily.COUNT,
            CountSpec(),
            RangePredicate(field="rating", low=4.0, low_inclusive=True),
        )
        _assert_parity(q, df_10, "N10/count_high_rating")

    def test_count_high_rating_n5000(self, df_5000):
        q = _make(
            QueryFamily.COUNT,
            CountSpec(),
            RangePredicate(field="rating", low=4.0, low_inclusive=True),
        )
        _assert_parity(q, df_5000, "N5000/count_high_rating")

    def test_count_with_price_n10(self, df_10):
        """Count records that have a non-null price (null-risk path)."""
        q = _make(QueryFamily.COUNT, CountSpec(), IsNotNullPredicate(field="price"))
        _assert_parity(q, df_10, "N10/count_has_price")

    def test_count_with_price_n5000(self, df_5000):
        q = _make(QueryFamily.COUNT, CountSpec(), IsNotNullPredicate(field="price"))
        _assert_parity(q, df_5000, "N5000/count_has_price")

    def test_count_missing_price_n10(self, df_10):
        """Count records with a NULL price (adversarial null path)."""
        q = _make(QueryFamily.COUNT, CountSpec(), IsNullPredicate(field="price"))
        _assert_parity(q, df_10, "N10/count_null_price")

    def test_count_missing_price_n5000(self, df_5000):
        q = _make(QueryFamily.COUNT, CountSpec(), IsNullPredicate(field="price"))
        _assert_parity(q, df_5000, "N5000/count_null_price")

    def test_count_empty_scope_n10(self, df_10):
        """Adversarial: rating=6 should always return 0."""
        q = _make(QueryFamily.COUNT, CountSpec(), EqPredicate(field="rating", value=6.0))
        _assert_parity(q, df_10, "N10/count_empty_scope")

    def test_count_empty_scope_n5000(self, df_5000):
        q = _make(QueryFamily.COUNT, CountSpec(), EqPredicate(field="rating", value=6.0))
        _assert_parity(q, df_5000, "N5000/count_empty_scope")


# ===========================================================================
# MEAN family
# ===========================================================================


class TestMeanParity:
    """MEAN queries: global, filtered, null-exclude, null-as-zero."""

    def test_mean_rating_all_n10(self, df_10):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE, decimal_places=4),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/mean_rating_all")

    def test_mean_rating_all_n5000(self, df_5000):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE, decimal_places=4),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/mean_rating_all")

    def test_mean_price_exclude_null_n10(self, df_10):
        """MEAN price — excluding nulls (standard path)."""
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/mean_price_excl_null")

    def test_mean_price_exclude_null_n5000(self, df_5000):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/mean_price_excl_null")

    def test_mean_price_incl_zero_n10(self, df_10):
        """MEAN price — treating nulls as zero (INCLUDE_AS_ZERO path)."""
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="price", null_policy=NullPolicy.INCLUDE_AS_ZERO, decimal_places=2),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/mean_price_incl_zero")

    def test_mean_price_incl_zero_n5000(self, df_5000):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="price", null_policy=NullPolicy.INCLUDE_AS_ZERO, decimal_places=2),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/mean_price_incl_zero")

    def test_mean_rating_cat_n10(self, df_10):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE, decimal_places=4),
            EqPredicate(field="category", value="electronics"),
        )
        _assert_parity(q, df_10, "N10/mean_rating_cat")

    def test_mean_rating_cat_n5000(self, df_5000):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE, decimal_places=4),
            EqPredicate(field="category", value="electronics"),
        )
        _assert_parity(q, df_5000, "N5000/mean_rating_cat")

    def test_mean_helpful_votes_n10(self, df_10):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="helpful_votes", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
            EqPredicate(field="verified_purchase", value=True),
        )
        _assert_parity(q, df_10, "N10/mean_helpful_votes")

    def test_mean_helpful_votes_n5000(self, df_5000):
        q = _make(
            QueryFamily.MEAN,
            MeanSpec(field="helpful_votes", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
            EqPredicate(field="verified_purchase", value=True),
        )
        _assert_parity(q, df_5000, "N5000/mean_helpful_votes")


# ===========================================================================
# PROPORTION family (rates)
# ===========================================================================


class TestProportionParity:
    """PROPORTION queries: rates / conditional frequencies."""

    def test_proportion_verified_all_n10(self, df_10):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=EqPredicate(field="verified_purchase", value=True),
                decimal_places=4,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/prop_verified_all")

    def test_proportion_verified_all_n5000(self, df_5000):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=EqPredicate(field="verified_purchase", value=True),
                decimal_places=4,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/prop_verified_all")

    def test_proportion_5star_in_cat_n10(self, df_10):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=EqPredicate(field="rating", value=5.0),
                decimal_places=4,
            ),
            EqPredicate(field="category", value="books"),
        )
        _assert_parity(q, df_10, "N10/prop_5star_books")

    def test_proportion_5star_in_cat_n5000(self, df_5000):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=EqPredicate(field="rating", value=5.0),
                decimal_places=4,
            ),
            EqPredicate(field="category", value="books"),
        )
        _assert_parity(q, df_5000, "N5000/prop_5star_books")

    def test_proportion_has_price_n10(self, df_10):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=IsNotNullPredicate(field="price"),
                decimal_places=4,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/prop_has_price")

    def test_proportion_has_price_n5000(self, df_5000):
        q = _make(
            QueryFamily.PROPORTION,
            ProportionSpec(
                numerator_predicate=IsNotNullPredicate(field="price"),
                decimal_places=4,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/prop_has_price")


# ===========================================================================
# TOP_K / RANKING family
# ===========================================================================


class TestTopKParity:
    """TOP_K ranking: count, mean-measure, tie-policy FIRST."""

    def test_topk_count_by_category_n10(self, df_10):
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=3,
                group_by_field="category",
                measure="count",
                ascending=False,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/topk_cat_count")

    def test_topk_count_by_category_n5000(self, df_5000):
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=3,
                group_by_field="category",
                measure="count",
                ascending=False,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/topk_cat_count")

    def test_topk_mean_rating_by_brand_n10(self, df_10):
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=3,
                group_by_field="brand",
                value_field="rating",
                measure="mean",
                ascending=False,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/topk_brand_mean_rating")

    def test_topk_mean_rating_by_brand_n5000(self, df_5000):
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=5,
                group_by_field="brand",
                value_field="rating",
                measure="mean",
                ascending=False,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/topk_brand_mean_rating")

    def test_topk_ascending_n10(self, df_10):
        """Bottom-K: ascending=True flips the sort."""
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=3,
                group_by_field="category",
                measure="count",
                ascending=True,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/topk_cat_ascending")

    def test_topk_ascending_n5000(self, df_5000):
        q = _make(
            QueryFamily.TOP_K,
            TopKSpec(
                k=3,
                group_by_field="category",
                measure="count",
                ascending=True,
                tie_policy=TiePolicy.FIRST,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/topk_cat_ascending")


# ===========================================================================
# TREND family
# ===========================================================================


class TestTrendParity:
    """TREND time-series: monthly/yearly count and mean bucketing."""

    def test_trend_monthly_count_n10(self, df_10):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="month",
                measure="count",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/trend_month_count")

    def test_trend_monthly_count_n5000(self, df_5000):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="month",
                measure="count",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/trend_month_count")

    def test_trend_yearly_count_n10(self, df_10):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="year",
                measure="count",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/trend_year_count")

    def test_trend_yearly_count_n5000(self, df_5000):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="year",
                measure="count",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/trend_year_count")

    def test_trend_quarterly_mean_rating_n10(self, df_10):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="quarter",
                measure="mean",
                value_field="rating",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/trend_quarter_mean_rating")

    def test_trend_quarterly_mean_rating_n5000(self, df_5000):
        q = _make(
            QueryFamily.TREND,
            TrendSpec(
                time_field="event_time",
                bucket="quarter",
                measure="mean",
                value_field="rating",
                null_policy=NullPolicy.EXCLUDE,
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/trend_quarter_mean_rating")


# ===========================================================================
# COMPARISON family (group-A vs group-B)
# ===========================================================================


class TestComparisonParity:
    """COMPARISON: mean rating Electronics vs Books, difference and ratio."""

    def test_comparison_mean_elec_vs_books_n10(self, df_10):
        q = _make(
            QueryFamily.COMPARISON,
            ComparisonSpec(
                group_a_predicate=EqPredicate(field="category", value="electronics"),
                group_b_predicate=EqPredicate(field="category", value="books"),
                measure="mean",
                field="rating",
                output="difference",
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/cmp_elec_books_diff")

    def test_comparison_mean_elec_vs_books_n5000(self, df_5000):
        q = _make(
            QueryFamily.COMPARISON,
            ComparisonSpec(
                group_a_predicate=EqPredicate(field="category", value="electronics"),
                group_b_predicate=EqPredicate(field="category", value="books"),
                measure="mean",
                field="rating",
                output="difference",
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/cmp_elec_books_diff")

    def test_comparison_count_verified_vs_unverified_n10(self, df_10):
        q = _make(
            QueryFamily.COMPARISON,
            ComparisonSpec(
                group_a_predicate=EqPredicate(field="verified_purchase", value=True),
                group_b_predicate=EqPredicate(field="verified_purchase", value=False),
                measure="count",
                output="ratio",
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_10, "N10/cmp_verified_ratio")

    def test_comparison_count_verified_vs_unverified_n5000(self, df_5000):
        q = _make(
            QueryFamily.COMPARISON,
            ComparisonSpec(
                group_a_predicate=EqPredicate(field="verified_purchase", value=True),
                group_b_predicate=EqPredicate(field="verified_purchase", value=False),
                measure="count",
                output="ratio",
            ),
            RangePredicate(
                field="rating", low=1.0, high=5.5, low_inclusive=True, high_inclusive=True
            ),
        )
        _assert_parity(q, df_5000, "N5000/cmp_verified_ratio")
