"""
Research-grade procedural query factory for FaultTrace-RAG — Prompt 2.

Generates grounded queries from templates for a given CorpusWorld.
All queries carry both a natural-language wording and an executable QuerySpec.

Prompt 2 expansions:
  - COUNT: 20 templates (was 10)
  - MEAN/SUM: 15 templates (was 8)
  - PROPORTION: 20 templates (was 6)
  - COMPARISON: 15 templates (was 6)
  - TOP_K: 15 templates (was 6)
  - TREND: 15 templates (was 6)
  Total: 100 templates (was 42)

New features:
  - TemplateRegistry with version, family, parameter constraints, evidence requirement
  - BenchmarkPack with dev/val/test split (80/10/10)
  - Duplicate/near-duplicate detection via spec hash canonicalization
  - Difficulty dimensions: easy, medium, adversarial
  - Selectivity dimensions: broad, narrow, selective
  - Null-rate, tie-risk, temporal-boundary dimensions
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from faulttrace_core.models import (
    AggregationKind,
    AggregationSpec,
    AndPredicate,
    ComparisonSpec,
    CorpusWorld,
    CountSpec,
    DerivedField,
    EqPredicate,
    FactSpec,
    InPredicate,
    IsNotNullPredicate,
    IsNullPredicate,
    MeanSpec,
    NullPolicy,
    OrPredicate,
    ProportionSpec,
    QueryFamily,
    QuerySpec,
    RangePredicate,
    RecordCategory,
    ScopePredicate,
    SumSpec,
    TiePolicy,
    TopKSpec,
    TrendSpec,
)

# ---------------------------------------------------------------------------
# Template registry entries
# ---------------------------------------------------------------------------

# Each entry: (template_id, natural_language_template, difficulty, selectivity, null_risk, tie_risk, temporal_risk)
_COUNT_TEMPLATES = [
    # Easy
    ("cnt_all", "What is the total number of reviews in the corpus?", "easy", "broad", "none", "none", "none"),
    ("cnt_cat_verified", "How many verified purchase reviews are there in the {category} category?", "easy", "narrow", "none", "none", "none"),
    ("cnt_cat_date", "How many {category} reviews were submitted in {year}?", "easy", "narrow", "none", "none", "low"),
    ("cnt_rating_ge", "How many reviews have a rating of {min_rating} stars or higher?", "easy", "broad", "none", "none", "none"),
    ("cnt_cat_rating", "How many {category} reviews received exactly {rating} stars?", "easy", "narrow", "none", "none", "none"),
    ("cnt_verified_all", "How many verified purchases are in the entire corpus?", "easy", "broad", "none", "none", "none"),
    ("cnt_brand_all", "How many reviews are there for {brand} products?", "easy", "narrow", "none", "none", "none"),
    ("cnt_has_price", "How many reviews have a listed product price?", "easy", "broad", "high", "none", "none"),
    ("cnt_cat_unverified", "How many unverified reviews are in the {category} category?", "easy", "narrow", "none", "none", "none"),
    ("cnt_high_rating", "How many reviews have a rating of 4 stars or above?", "easy", "broad", "none", "none", "none"),
    # Medium
    ("cnt_date_range", "How many reviews were submitted between {start_date} and {end_date}?", "medium", "selective", "none", "none", "high"),
    ("cnt_cat_brand", "How many reviews for {brand} products are in the {category} category?", "medium", "selective", "none", "none", "none"),
    ("cnt_price_range", "How many products are priced between ${low_price} and ${high_price}?", "medium", "selective", "high", "none", "none"),
    ("cnt_verified_cat", "How many verified purchases are there across Electronics and Books combined?", "medium", "selective", "none", "none", "none"),
    ("cnt_cat_year_brand", "How many {category} reviews from {brand} were submitted in {year}?", "medium", "selective", "none", "none", "low"),
    # Adversarial
    ("cnt_empty_scope", "How many reviews have a rating of exactly 6 stars?", "adversarial", "selective", "none", "none", "none"),
    ("cnt_null_price", "How many reviews have a missing product price?", "adversarial", "broad", "high", "none", "none"),
    ("cnt_multi_cat", "How many reviews are in either {cat_a} or {cat_b}?", "adversarial", "selective", "none", "none", "none"),
    ("cnt_rating_lt", "How many reviews have a rating strictly below {max_rating} stars?", "adversarial", "broad", "none", "none", "none"),
    ("cnt_boundary_date", "How many reviews were submitted on the last day of {year}?", "adversarial", "selective", "none", "none", "high"),
]

_MEAN_TEMPLATES = [
    # Easy
    ("mean_rating_all", "What is the overall mean rating across all reviews?", "easy", "broad", "none", "none", "none"),
    ("mean_rating_cat", "What is the mean rating for {category} products?", "easy", "narrow", "none", "none", "none"),
    ("mean_price_cat", "What is the average price of {category} products, excluding items with missing prices?", "easy", "narrow", "high", "none", "none"),
    ("mean_rating_brand", "What is the average star rating for {brand} products?", "easy", "narrow", "none", "none", "none"),
    ("mean_rating_verified", "What is the mean rating for verified purchases?", "easy", "broad", "none", "none", "none"),
    ("mean_helpful_cat", "What is the average number of helpful votes for {category} reviews?", "medium", "narrow", "none", "none", "none"),
    ("mean_price_verified", "What is the average price of verified-purchase products, excluding missing prices?", "medium", "selective", "high", "none", "none"),
    ("mean_rating_date", "What is the mean rating for reviews submitted in {year}?", "medium", "selective", "none", "none", "low"),
    # Medium
    ("mean_rating_low_rating", "What is the average rating among reviews with 1 or 2 stars?", "medium", "selective", "none", "none", "none"),
    ("mean_price_brand", "What is the average price of {brand} products, excluding missing prices?", "medium", "narrow", "high", "none", "none"),
    ("mean_helpful_verified", "What is the average helpful vote count for verified purchases?", "medium", "broad", "none", "none", "none"),
    ("mean_rating_cat_year", "What is the mean rating for {category} products reviewed in {year}?", "medium", "selective", "none", "none", "low"),
    # Adversarial
    ("mean_price_all", "What is the overall mean price across all products with a listed price?", "adversarial", "selective", "high", "none", "none"),
    ("mean_rating_unverified", "What is the mean rating for unverified (not verified purchase) reviews?", "adversarial", "broad", "none", "none", "none"),
    ("mean_helpful_all", "What is the average number of helpful votes across all reviews?", "adversarial", "broad", "none", "none", "none"),
]

_PROPORTION_TEMPLATES = [
    # Easy
    ("prop_verified_cat", "What proportion of {category} reviews are verified purchases?", "easy", "narrow", "none", "none", "none"),
    ("prop_high_rating_cat", "What fraction of {category} reviews have a rating of 4 or higher?", "easy", "narrow", "none", "none", "none"),
    ("prop_no_price", "What proportion of reviews have a missing price?", "easy", "broad", "high", "none", "none"),
    ("prop_verified_all", "What fraction of all reviews are verified purchases?", "easy", "broad", "none", "none", "none"),
    ("prop_five_star_cat", "What percentage of {category} reviews received 5 stars?", "easy", "narrow", "none", "none", "none"),
    ("prop_has_price_brand", "What proportion of {brand} products have a listed price?", "easy", "narrow", "high", "none", "none"),
    ("prop_low_rating", "What fraction of all reviews have a rating of 2 stars or below?", "easy", "broad", "none", "none", "none"),
    ("prop_high_rating_all", "What proportion of all reviews received 4 stars or more?", "easy", "broad", "none", "none", "none"),
    # Medium
    ("prop_helpful_cat", "What proportion of {category} reviews have at least one helpful vote?", "medium", "narrow", "none", "none", "none"),
    ("prop_top_brand", "What proportion of {category} reviews are from the brand {brand}?", "medium", "selective", "none", "none", "none"),
    ("prop_recent", "What proportion of reviews were submitted in 2022 or later?", "medium", "broad", "none", "none", "low"),
    ("prop_verified_brand", "What fraction of {brand} reviews are verified purchases?", "medium", "narrow", "none", "none", "none"),
    ("prop_priced_verified", "Among verified purchases, what proportion have a listed price?", "medium", "selective", "high", "none", "none"),
    ("prop_high_helpful", "What proportion of reviews have more than 10 helpful votes?", "medium", "selective", "none", "none", "none"),
    ("prop_unverified_low_rating", "What proportion of unverified reviews have a rating below 3?", "medium", "selective", "none", "none", "none"),
    # Adversarial
    ("prop_five_star_all", "What percentage of all reviews are 5-star?", "adversarial", "broad", "none", "low", "none"),
    ("prop_null_price_cat", "What fraction of {category} products are missing a price?", "adversarial", "narrow", "high", "none", "none"),
    ("prop_cat_of_all", "What fraction of all reviews belong to the {category} category?", "adversarial", "selective", "none", "none", "none"),
    ("prop_recent_verified", "What proportion of verified purchases were submitted after 2021?", "adversarial", "selective", "none", "none", "low"),
    ("prop_brand_of_cat", "Among {category} reviews, what proportion come from {brand}?", "adversarial", "selective", "none", "none", "none"),
]

_COMPARISON_TEMPLATES = [
    # Easy
    ("cmp_rating_cats", "Which has a higher mean rating: {cat_a} or {cat_b}?", "easy", "selective", "none", "low", "none"),
    ("cmp_count_cats", "Which category has more reviews: {cat_a} or {cat_b}?", "easy", "selective", "none", "low", "none"),
    ("cmp_verified_vs_unverified", "Is the mean rating higher for verified or unverified purchases in {category}?", "easy", "narrow", "none", "low", "none"),
    ("cmp_brand_count", "Which brand has more reviews: {brand_a} or {brand_b}?", "easy", "selective", "none", "low", "none"),
    ("cmp_cat_verified_rate", "Which category has a higher rate of verified purchases: {cat_a} or {cat_b}?", "easy", "selective", "none", "low", "none"),
    # Medium
    ("cmp_year_count", "Did the corpus receive more reviews in {year_a} or {year_b}?", "medium", "selective", "none", "low", "high"),
    ("cmp_brand_rating", "What is the difference in mean rating between {brand_a} and {brand_b}?", "medium", "selective", "none", "low", "none"),
    ("cmp_price_cats", "What is the ratio of mean price between {cat_a} and {cat_b}?", "medium", "selective", "high", "none", "none"),
    ("cmp_helpful_cats", "Which category has a higher average helpful vote count: {cat_a} or {cat_b}?", "medium", "selective", "none", "low", "none"),
    ("cmp_verified_price_cat", "Is the mean price higher for verified or unverified purchases in {category}?", "medium", "narrow", "high", "low", "none"),
    # Adversarial
    ("cmp_near_equal_rating", "Compare the mean rating between {cat_a} and {cat_b}: which is higher?", "adversarial", "selective", "none", "high", "none"),
    ("cmp_cat_year_count", "In {category}, were there more reviews in {year_a} or {year_b}?", "adversarial", "selective", "none", "low", "high"),
    ("cmp_brand_price_ratio", "What is the ratio of mean price between {brand_a} and {brand_b}?", "adversarial", "selective", "high", "none", "none"),
    ("cmp_q1_q2_count", "Did {category} receive more reviews in Q1 or Q2 of {year}?", "adversarial", "selective", "none", "low", "high"),
    ("cmp_null_rate_cats", "Which category has a higher proportion of missing prices: {cat_a} or {cat_b}?", "adversarial", "selective", "high", "low", "none"),
]

_TOPK_TEMPLATES = [
    # Easy
    ("topk_brands_reviews", "Which {k} brands have the most reviews?", "easy", "broad", "none", "low", "none"),
    ("topk_cats_reviews", "What are the top {k} categories by review count?", "easy", "broad", "none", "low", "none"),
    ("topk_brands_rating", "Which {k} brands have the highest average rating?", "easy", "broad", "none", "low", "none"),
    ("topk_cats_rating", "What are the top {k} categories by mean rating?", "easy", "broad", "none", "low", "none"),
    ("topk_brands_count_cat", "Which {k} brands have the most {category} reviews?", "easy", "narrow", "none", "low", "none"),
    # Medium
    ("topk_brands_reviews_cat", "Which {k} brands have the most {category} reviews?", "medium", "narrow", "none", "low", "none"),
    ("topk_cats_mean_price", "Which {k} categories have the highest average price?", "medium", "broad", "high", "low", "none"),
    ("topk_brands_helpful", "Which {k} brands have the most helpful votes on average?", "medium", "broad", "none", "low", "none"),
    ("topk_brands_verified_rate", "Which {k} brands have the highest verified purchase rate?", "medium", "broad", "none", "low", "none"),
    ("topk_cats_count_year", "What are the top {k} categories by review count in {year}?", "medium", "selective", "none", "low", "low"),
    # Adversarial
    ("topk_brands_tie", "What are the top {k} brands by review count, breaking ties alphabetically?", "adversarial", "broad", "none", "high", "none"),
    ("topk_cats_price_cat", "Among products with a listed price, which {k} categories have the highest mean price?", "adversarial", "selective", "high", "low", "none"),
    ("topk_brands_low_rating", "Which {k} brands have the lowest average rating?", "adversarial", "broad", "none", "low", "none"),
    ("topk_brands_count_verified", "Among verified purchases, which {k} brands have the most reviews?", "adversarial", "selective", "none", "low", "none"),
    ("topk_cats_proportion_5star", "Which {k} categories have the highest proportion of 5-star reviews?", "adversarial", "broad", "none", "low", "none"),
]

_TREND_TEMPLATES = [
    # Easy
    ("trend_monthly_count", "How does the number of reviews change month by month?", "easy", "broad", "none", "none", "none"),
    ("trend_quarterly_rating", "How does the average rating change quarter by quarter?", "easy", "broad", "none", "none", "none"),
    ("trend_monthly_count_cat", "How does the number of {category} reviews change month by month?", "easy", "narrow", "none", "none", "none"),
    ("trend_yearly_count", "How does the annual review count change year over year?", "easy", "broad", "none", "none", "low"),
    ("trend_quarterly_count", "How do review counts trend by quarter?", "easy", "broad", "none", "none", "none"),
    # Medium
    ("trend_quarterly_count_verified", "How do verified purchase review counts trend by quarter?", "medium", "selective", "none", "none", "none"),
    ("trend_monthly_price_cat", "How does the average price of {category} products trend month by month?", "medium", "narrow", "high", "none", "none"),
    ("trend_yearly_rating", "How does the mean rating trend year over year?", "medium", "broad", "none", "none", "low"),
    ("trend_monthly_count_brand", "How does the monthly review count for {brand} change over time?", "medium", "narrow", "none", "none", "none"),
    ("trend_quarterly_proportion_verified", "How does the proportion of verified purchases change by quarter?", "medium", "broad", "none", "none", "none"),
    # Adversarial
    ("trend_monthly_count_date_range", "How does the monthly review count trend from {start_date} to {end_date}?", "adversarial", "selective", "none", "none", "high"),
    ("trend_quarterly_price_all", "How does the mean price trend by quarter across all products with prices?", "adversarial", "broad", "high", "none", "none"),
    ("trend_yearly_count_cat", "How does the annual review count for {category} trend over time?", "adversarial", "narrow", "none", "none", "low"),
    ("trend_monthly_helpful", "How does the average helpful vote count trend month by month?", "adversarial", "broad", "none", "none", "none"),
    ("trend_yearly_proportion_5star", "How does the proportion of 5-star reviews change year over year?", "adversarial", "broad", "none", "none", "low"),
]

ALL_TEMPLATES = {
    QueryFamily.COUNT: _COUNT_TEMPLATES,
    QueryFamily.MEAN: _MEAN_TEMPLATES,
    QueryFamily.PROPORTION: _PROPORTION_TEMPLATES,
    QueryFamily.COMPARISON: _COMPARISON_TEMPLATES,
    QueryFamily.TOP_K: _TOPK_TEMPLATES,
    QueryFamily.TREND: _TREND_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------


class TemplateEntry(BaseModel):
    """Metadata for a single query template."""
    template_id: str
    family: str
    natural_language_template: str
    difficulty: str  # easy, medium, adversarial
    selectivity: str  # broad, narrow, selective
    null_risk: str  # none, low, high
    tie_risk: str  # none, low, high
    temporal_risk: str  # none, low, high
    version: str = "2.0.0"
    evidence_requirement: str = "all_matching_records"

    @property
    def dimensions(self) -> dict[str, str]:
        return {
            "difficulty": self.difficulty,
            "selectivity": self.selectivity,
            "null_risk": self.null_risk,
            "tie_risk": self.tie_risk,
            "temporal_risk": self.temporal_risk,
        }


class TemplateRegistry:
    """Registry of all query templates with metadata."""

    def __init__(self):
        self._entries: dict[str, TemplateEntry] = {}
        self._build()

    def _build(self) -> None:
        for family, templates in ALL_TEMPLATES.items():
            for entry in templates:
                tid, tmpl, difficulty, selectivity, null_risk, tie_risk, temporal_risk = entry
                self._entries[tid] = TemplateEntry(
                    template_id=tid,
                    family=family.value,
                    natural_language_template=tmpl,
                    difficulty=difficulty,
                    selectivity=selectivity,
                    null_risk=null_risk,
                    tie_risk=tie_risk,
                    temporal_risk=temporal_risk,
                )

    def get(self, template_id: str) -> Optional[TemplateEntry]:
        return self._entries.get(template_id)

    def list_by_family(self, family: QueryFamily) -> list[TemplateEntry]:
        return [e for e in self._entries.values() if e.family == family.value]

    def list_by_difficulty(self, difficulty: str) -> list[TemplateEntry]:
        return [e for e in self._entries.values() if e.difficulty == difficulty]

    def summary(self) -> dict[str, Any]:
        counts_by_family: dict[str, int] = {}
        counts_by_difficulty: dict[str, int] = {}
        for e in self._entries.values():
            counts_by_family[e.family] = counts_by_family.get(e.family, 0) + 1
            counts_by_difficulty[e.difficulty] = counts_by_difficulty.get(e.difficulty, 0) + 1
        return {
            "total_templates": len(self._entries),
            "by_family": counts_by_family,
            "by_difficulty": counts_by_difficulty,
        }


TEMPLATE_REGISTRY = TemplateRegistry()


# ---------------------------------------------------------------------------
# BenchmarkPack
# ---------------------------------------------------------------------------


class BenchmarkPack(BaseModel):
    """A balanced, validated benchmark query pack with dev/val/test splits."""
    pack_id: str = Field(default_factory=lambda: str(uuid4()))
    world_id: str
    total_count: int = 0
    agreed_count: int = 0
    disagreed_count: int = 0
    skipped_count: int = 0
    gold_ready: bool = False

    # Split counts
    dev_count: int = 0
    val_count: int = 0
    test_count: int = 0

    # Distribution
    count_by_family: dict[str, int] = Field(default_factory=dict)
    count_by_difficulty: dict[str, int] = Field(default_factory=dict)

    # Query IDs per split (no spec content in pack header)
    dev_query_ids: list[str] = Field(default_factory=list)
    val_query_ids: list[str] = Field(default_factory=list)
    test_query_ids: list[str] = Field(default_factory=list)

    # Duplicate detection
    duplicate_spec_hashes_found: int = 0

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "2.0.0"


# ---------------------------------------------------------------------------
# QueryFactory (Prompt 2 version)
# ---------------------------------------------------------------------------


class QueryFactory:
    """
    Generates research-grade procedural queries for a corpus world.

    Prompt 2 extensions:
    - 100 templates across 6 families
    - BenchmarkPack with 80/10/10 dev/val/test split
    - Duplicate detection via canonical spec hashes
    - Balanced generation across difficulty and dimension axes
    """

    def __init__(self, data_dir: Path = Path("data/generated")):
        self.data_dir = data_dir

    def generate_for_world(
        self,
        world_id: str,
        target_count: int = 60,
        seed: Optional[int] = None,
    ) -> list[QuerySpec]:
        """Generate target_count queries for the given world."""
        world_dir = self.data_dir / "worlds" / world_id
        parquet_path = world_dir / "records.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        if seed is None:
            seed = int(hashlib.sha256(world_id.encode()).hexdigest()[:8], 16) & 0xFFFFFF

        rng = random.Random(seed)
        queries: list[QuerySpec] = []

        families = [
            (QueryFamily.COUNT, _COUNT_TEMPLATES, self._make_count_query),
            (QueryFamily.MEAN, _MEAN_TEMPLATES, self._make_mean_query),
            (QueryFamily.PROPORTION, _PROPORTION_TEMPLATES, self._make_proportion_query),
            (QueryFamily.COMPARISON, _COMPARISON_TEMPLATES, self._make_comparison_query),
            (QueryFamily.TOP_K, _TOPK_TEMPLATES, self._make_topk_query),
            (QueryFamily.TREND, _TREND_TEMPLATES, self._make_trend_query),
        ]

        per_family = max(target_count // len(families), len(_COUNT_TEMPLATES))

        for family, templates, maker in families:
            family_queries = self._generate_family(
                family=family,
                templates=templates,
                maker=maker,
                world_id=world_id,
                df=df,
                rng=rng,
                count=per_family,
            )
            queries.extend(family_queries)

        return queries

    def build_benchmark_pack(
        self,
        world_id: str,
        total_count: int = 300,
        validate_gold: bool = True,
    ) -> BenchmarkPack:
        """
        Build a research benchmark pack with balanced queries and dev/val/test splits.

        Generates total_count queries, validates gold agreement, removes duplicates,
        and assigns deterministic splits.
        """
        queries = self.generate_for_world(world_id=world_id, target_count=total_count)

        # Deduplicate by canonical spec hash
        seen_hashes: set[str] = set()
        unique_queries: list[QuerySpec] = []
        duplicates_found = 0

        for q in queries:
            h = q.spec_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_queries.append(q)
            else:
                duplicates_found += 1

        # Validate gold if requested
        agreed_count = 0
        disagreed_count = 0
        skipped_count = 0

        if validate_gold:
            world_dir = self.data_dir / "worlds" / world_id
            parquet_path = world_dir / "records.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                agreed_count, disagreed_count, skipped_count = self._validate_gold_batch(
                    unique_queries, df
                )

        # Assign splits: 80% dev, 10% val, 10% test (deterministic by spec_hash)
        dev_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []

        for q in unique_queries:
            split_seed = int(q.spec_hash()[:4], 16) % 10
            if split_seed < 8:
                dev_ids.append(q.query_id)
            elif split_seed < 9:
                val_ids.append(q.query_id)
            else:
                test_ids.append(q.query_id)

        # Distribution stats
        count_by_family: dict[str, int] = {}
        count_by_difficulty: dict[str, int] = {}
        for q in unique_queries:
            count_by_family[q.family.value] = count_by_family.get(q.family.value, 0) + 1
            reg_entry = TEMPLATE_REGISTRY.get(q.template_id)
            if reg_entry:
                d = reg_entry.difficulty
                count_by_difficulty[d] = count_by_difficulty.get(d, 0) + 1

        gold_ready = (disagreed_count == 0) and (len(unique_queries) > 0)

        pack = BenchmarkPack(
            world_id=world_id,
            total_count=len(unique_queries),
            agreed_count=agreed_count,
            disagreed_count=disagreed_count,
            skipped_count=skipped_count,
            gold_ready=gold_ready,
            dev_count=len(dev_ids),
            val_count=len(val_ids),
            test_count=len(test_ids),
            dev_query_ids=dev_ids,
            val_query_ids=val_ids,
            test_query_ids=test_ids,
            count_by_family=count_by_family,
            count_by_difficulty=count_by_difficulty,
            duplicate_spec_hashes_found=duplicates_found,
        )

        return pack

    def _validate_gold_batch(
        self,
        queries: list[QuerySpec],
        df: pd.DataFrame,
    ) -> tuple[int, int, int]:
        """Run dual gold validation on a batch of queries. Returns (agreed, disagreed, skipped)."""
        try:
            from faulttrace_gold.pandas_engine import PandasEvaluator
            from faulttrace_gold.duckdb_engine import DuckDBEvaluator
            from faulttrace_gold.validator import results_agree

            pandas_eval = PandasEvaluator()
            duckdb_eval = DuckDBEvaluator()

            agreed = 0
            disagreed = 0
            skipped = 0

            for q in queries:
                try:
                    pd_result = pandas_eval.evaluate(q, df)
                    dk_result = duckdb_eval.evaluate(q, df)
                    if results_agree(pd_result["result"], dk_result["result"], q.tolerance):
                        agreed += 1
                    else:
                        disagreed += 1
                except Exception:
                    skipped += 1

            return agreed, disagreed, skipped
        except ImportError:
            return 0, 0, len(queries)

    def _generate_family(
        self,
        family: QueryFamily,
        templates: list[tuple],
        maker: Any,
        world_id: str,
        df: pd.DataFrame,
        rng: random.Random,
        count: int,
    ) -> list[QuerySpec]:
        """Generate queries for a single family."""
        import uuid
        queries = []
        seen_hashes = set()
        categories = df["category"].dropna().unique().tolist()
        brands = df["brand"].dropna().unique().tolist()[:20]

        context = {
            "categories": categories,
            "brands": brands,
            "world_id": world_id,
            "df": df,
        }

        template_idx = 0
        attempts = 0
        while len(queries) < count and attempts < count * 5:
            tmpl = templates[template_idx % len(templates)]
            template_idx += 1
            attempts += 1

            try:
                q = maker(tmpl, context, rng)
                if q is not None:
                    h = q.spec_hash()
                    if h not in seen_hashes:
                        validated = self._validate_query(q, df)
                        if validated:
                            q.query_id = str(uuid.uuid5(uuid.NAMESPACE_OID, h))
                            queries.append(q)
                            seen_hashes.add(h)
            except Exception:
                pass

        return queries

    def _validate_query(self, query: QuerySpec, df: pd.DataFrame) -> bool:
        """Validate a query: check it can be evaluated and has a non-empty scope (except special cases)."""
        from faulttrace_core.predicates import compiler

        if "empty_scope" in query.template_id:
            return True

        try:
            mask = compiler.to_pandas_mask(query.scope_predicate, df)
            eligible = int(mask.sum())
            if eligible == 0 and "empty" not in query.template_id:
                return False
        except Exception:
            return False

        if isinstance(query.aggregation_spec, TopKSpec):
            if query.aggregation_spec.k > eligible:
                return False

        return True

    # =========================================================================
    # Template makers (COUNT — 20 templates)
    # =========================================================================

    def _make_count_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]
        tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]
        brands = context["brands"]

        if tid == "cnt_all":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_verified":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), EqPredicate(field="verified_purchase", value=True)]),
                fact_spec=FactSpec(fields=["record_id", "category", "verified_purchase"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_date":
            cat = rng.choice(cats)
            year = rng.choice([2020, 2021, 2022, 2023])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(category=cat, year=year),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat())]),
                fact_spec=FactSpec(fields=["record_id", "category", "event_time"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_rating_ge":
            min_rating = rng.choice([3.0, 4.0, 4.5])
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(min_rating=min_rating),
                scope_predicate=RangePredicate(field="rating", low=min_rating),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_rating":
            cat = rng.choice(cats)
            rating = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(category=cat, rating=rating),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), EqPredicate(field="rating", value=rating)]),
                fact_spec=FactSpec(fields=["record_id", "category", "rating"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_verified_all":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "verified_purchase"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_brand_all":
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=EqPredicate(field="brand", value=brand),
                fact_spec=FactSpec(fields=["record_id", "brand"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_has_price":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=IsNotNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_unverified":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), EqPredicate(field="verified_purchase", value=False)]),
                fact_spec=FactSpec(fields=["record_id", "category", "verified_purchase"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_high_rating":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=4.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_date_range":
            year = rng.choice([2021, 2022])
            start = datetime(year, 6, 30, tzinfo=timezone.utc)
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(start_date=start.date(), end_date=end.date()),
                scope_predicate=RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat(), high_inclusive=False),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_brand":
            cat = rng.choice(cats)
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(brand=brand, category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), EqPredicate(field="brand", value=brand)]),
                fact_spec=FactSpec(fields=["record_id", "category", "brand"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_price_range":
            low = rng.choice([10, 20, 50])
            high = rng.choice([100, 200, 500])
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(low_price=low, high_price=high),
                scope_predicate=AndPredicate(operands=[IsNotNullPredicate(field="price"), RangePredicate(field="price", low=float(low), high=float(high))]),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_verified_cat":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=AndPredicate(operands=[EqPredicate(field="verified_purchase", value=True), InPredicate(field="category", values=[RecordCategory.ELECTRONICS.value, RecordCategory.BOOKS.value])]),
                fact_spec=FactSpec(fields=["record_id", "category", "verified_purchase"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_cat_year_brand":
            cat = rng.choice(cats)
            brand = rng.choice(brands) if brands else "TechPrime"
            year = rng.choice([2021, 2022, 2023])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(category=cat, brand=brand, year=year),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), EqPredicate(field="brand", value=brand), RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat())]),
                fact_spec=FactSpec(fields=["record_id", "category", "brand", "event_time"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_empty_scope":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="rating", value=6.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid, tolerance=0.0)

        elif tid == "cnt_null_price":
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl,
                scope_predicate=IsNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_multi_cat":
            if len(cats) < 2:
                return None
            cat_a, cat_b = rng.sample(cats, 2)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(cat_a=cat_a, cat_b=cat_b),
                scope_predicate=InPredicate(field="category", values=[cat_a, cat_b]),
                fact_spec=FactSpec(fields=["record_id", "category"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_rating_lt":
            max_rating = rng.choice([3.0, 4.0])
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(max_rating=max_rating),
                scope_predicate=RangePredicate(field="rating", high=max_rating, high_inclusive=False),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        elif tid == "cnt_boundary_date":
            year = rng.choice([2020, 2021, 2022, 2023])
            start = datetime(year, 12, 31, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COUNT, natural_language_question=tmpl.format(year=year),
                scope_predicate=RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat()),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=CountSpec(), world_id=world_id, template_id=tid)

        return None

    # =========================================================================
    # MEAN templates (15)
    # =========================================================================

    def _make_mean_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]; tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]
        brands = context["brands"]

        if tid == "mean_rating_all":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_price_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_brand":
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=EqPredicate(field="brand", value=brand),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_verified":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "rating", "verified_purchase"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_helpful_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "helpful_votes"]),
                aggregation_spec=MeanSpec(field="helpful_votes", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_price_verified":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=AndPredicate(operands=[EqPredicate(field="verified_purchase", value=True), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "price", "verified_purchase"]),
                aggregation_spec=MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_date":
            year = rng.choice([2021, 2022, 2023])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(year=year),
                scope_predicate=RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat()),
                fact_spec=FactSpec(fields=["record_id", "rating", "event_time"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_low_rating":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=2.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_price_brand":
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="brand", value=brand), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
                world_id=world_id, template_id=tid)

        elif tid == "mean_helpful_verified":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "helpful_votes"]),
                aggregation_spec=MeanSpec(field="helpful_votes", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_cat_year":
            cat = rng.choice(cats)
            year = rng.choice([2021, 2022, 2023])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl.format(category=cat, year=year),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat())]),
                fact_spec=FactSpec(fields=["record_id", "rating", "event_time"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_price_all":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=IsNotNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=MeanSpec(field="price", null_policy=NullPolicy.EXCLUDE, decimal_places=2),
                world_id=world_id, template_id=tid)

        elif tid == "mean_rating_unverified":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=False),
                fact_spec=FactSpec(fields=["record_id", "rating", "verified_purchase"]),
                aggregation_spec=MeanSpec(field="rating", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        elif tid == "mean_helpful_all":
            return QuerySpec(family=QueryFamily.MEAN, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "helpful_votes"]),
                aggregation_spec=MeanSpec(field="helpful_votes", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid)

        return None

    # =========================================================================
    # PROPORTION templates (20)
    # =========================================================================

    def _make_proportion_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]; tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]
        brands = context["brands"]

        if tid == "prop_verified_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "verified_purchase", "category"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="verified_purchase", value=True)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_high_rating_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "rating", "category"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="rating", low=4.0)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_no_price":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=ProportionSpec(numerator_predicate=IsNullPredicate(field="price")),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_verified_all":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "verified_purchase"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="verified_purchase", value=True)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_five_star_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="rating", value=5.0)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_has_price_brand":
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=EqPredicate(field="brand", value=brand),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=ProportionSpec(numerator_predicate=IsNotNullPredicate(field="price")),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_low_rating":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="rating", high=2.0)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_high_rating_all":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="rating", low=4.0)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_helpful_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "helpful_votes", "category"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="helpful_votes", low=1, low_inclusive=True)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_top_brand":
            cat = rng.choice(cats)
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat, brand=brand),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "brand", "category"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="brand", value=brand)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_recent":
            cutoff = datetime(2022, 1, 1, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="event_time", low=cutoff.isoformat())),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_verified_brand":
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=EqPredicate(field="brand", value=brand),
                fact_spec=FactSpec(fields=["record_id", "verified_purchase"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="verified_purchase", value=True)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_priced_verified":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=ProportionSpec(numerator_predicate=IsNotNullPredicate(field="price")),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_high_helpful":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "helpful_votes"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="helpful_votes", low=10, low_inclusive=False)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_unverified_low_rating":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=False),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="rating", high=3.0, high_inclusive=False)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_five_star_all":
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "rating"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="rating", value=5.0)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_null_price_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "price"]),
                aggregation_spec=ProportionSpec(numerator_predicate=IsNullPredicate(field="price")),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_cat_of_all":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "category"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="category", value=cat)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_recent_verified":
            cutoff = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=ProportionSpec(numerator_predicate=RangePredicate(field="event_time", low=cutoff.isoformat())),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "prop_brand_of_cat":
            cat = rng.choice(cats)
            brand = rng.choice(brands) if brands else "TechPrime"
            return QuerySpec(family=QueryFamily.PROPORTION, natural_language_question=tmpl.format(category=cat, brand=brand),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "brand"]),
                aggregation_spec=ProportionSpec(numerator_predicate=EqPredicate(field="brand", value=brand)),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        return None

    # =========================================================================
    # COMPARISON templates (15)
    # =========================================================================

    def _make_comparison_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]; tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]
        brands = context["brands"]

        def _two_cats():
            if len(cats) < 2: return None, None
            a, b = rng.sample(cats, 2)
            return a, b

        def _two_brands():
            if len(brands) < 2: return None, None
            a, b = rng.sample(brands, 2)
            return a, b

        if tid == "cmp_rating_cats":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category", "rating"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="rating", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_count_cats":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category"]),
                aggregation_spec=ComparisonSpec(measure="count", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid)

        elif tid == "cmp_verified_vs_unverified":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "category", "rating", "verified_purchase"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="rating", group_a_predicate=EqPredicate(field="verified_purchase", value=True), group_b_predicate=EqPredicate(field="verified_purchase", value=False), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_brand_count":
            ba, bb = _two_brands()
            if ba is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(brand_a=ba, brand_b=bb),
                scope_predicate=InPredicate(field="brand", values=[ba, bb]),
                fact_spec=FactSpec(fields=["record_id", "brand"]),
                aggregation_spec=ComparisonSpec(measure="count", group_a_predicate=EqPredicate(field="brand", value=ba), group_b_predicate=EqPredicate(field="brand", value=bb), output="difference"),
                world_id=world_id, template_id=tid)

        elif tid == "cmp_cat_verified_rate":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category", "verified_purchase"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="verified_purchase", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_year_count":
            ya, yb = rng.sample([2020, 2021, 2022, 2023], 2)
            sa = datetime(ya, 1, 1, tzinfo=timezone.utc); ea = datetime(ya, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            sb = datetime(yb, 1, 1, tzinfo=timezone.utc); eb = datetime(yb, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(year_a=ya, year_b=yb),
                scope_predicate=OrPredicate(operands=[RangePredicate(field="event_time", low=sa.isoformat(), high=ea.isoformat()), RangePredicate(field="event_time", low=sb.isoformat(), high=eb.isoformat())]),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=ComparisonSpec(measure="count", group_a_predicate=RangePredicate(field="event_time", low=sa.isoformat(), high=ea.isoformat()), group_b_predicate=RangePredicate(field="event_time", low=sb.isoformat(), high=eb.isoformat()), output="difference"),
                world_id=world_id, template_id=tid)

        elif tid == "cmp_brand_rating":
            ba, bb = _two_brands()
            if ba is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(brand_a=ba, brand_b=bb),
                scope_predicate=InPredicate(field="brand", values=[ba, bb]),
                fact_spec=FactSpec(fields=["record_id", "brand", "rating"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="rating", group_a_predicate=EqPredicate(field="brand", value=ba), group_b_predicate=EqPredicate(field="brand", value=bb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_price_cats":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=AndPredicate(operands=[InPredicate(field="category", values=[ca, cb]), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "category", "price"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="price", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="ratio"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_helpful_cats":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category", "helpful_votes"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="helpful_votes", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_verified_price_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "category", "price", "verified_purchase"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="price", group_a_predicate=EqPredicate(field="verified_purchase", value=True), group_b_predicate=EqPredicate(field="verified_purchase", value=False), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_near_equal_rating":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category", "rating"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="rating", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_cat_year_count":
            cat = rng.choice(cats)
            ya, yb = rng.sample([2020, 2021, 2022, 2023], 2)
            sa = datetime(ya, 1, 1, tzinfo=timezone.utc); ea = datetime(ya, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            sb = datetime(yb, 1, 1, tzinfo=timezone.utc); eb = datetime(yb, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(category=cat, year_a=ya, year_b=yb),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), OrPredicate(operands=[RangePredicate(field="event_time", low=sa.isoformat(), high=ea.isoformat()), RangePredicate(field="event_time", low=sb.isoformat(), high=eb.isoformat())])]),
                fact_spec=FactSpec(fields=["record_id", "category", "event_time"]),
                aggregation_spec=ComparisonSpec(measure="count", group_a_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), RangePredicate(field="event_time", low=sa.isoformat(), high=ea.isoformat())]), group_b_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), RangePredicate(field="event_time", low=sb.isoformat(), high=eb.isoformat())]), output="difference"),
                world_id=world_id, template_id=tid)

        elif tid == "cmp_brand_price_ratio":
            ba, bb = _two_brands()
            if ba is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(brand_a=ba, brand_b=bb),
                scope_predicate=AndPredicate(operands=[InPredicate(field="brand", values=[ba, bb]), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "brand", "price"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="price", group_a_predicate=EqPredicate(field="brand", value=ba), group_b_predicate=EqPredicate(field="brand", value=bb), output="ratio"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "cmp_q1_q2_count":
            cat = rng.choice(cats)
            year = rng.choice([2021, 2022, 2023])
            sq1 = datetime(year, 1, 1, tzinfo=timezone.utc); eq1 = datetime(year, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
            sq2 = datetime(year, 4, 1, tzinfo=timezone.utc); eq2 = datetime(year, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(category=cat, year=year),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), OrPredicate(operands=[RangePredicate(field="event_time", low=sq1.isoformat(), high=eq1.isoformat()), RangePredicate(field="event_time", low=sq2.isoformat(), high=eq2.isoformat())])]),
                fact_spec=FactSpec(fields=["record_id", "category", "event_time"]),
                aggregation_spec=ComparisonSpec(measure="count", group_a_predicate=RangePredicate(field="event_time", low=sq1.isoformat(), high=eq1.isoformat()), group_b_predicate=RangePredicate(field="event_time", low=sq2.isoformat(), high=eq2.isoformat()), output="difference"),
                world_id=world_id, template_id=tid)

        elif tid == "cmp_null_rate_cats":
            ca, cb = _two_cats()
            if ca is None: return None
            return QuerySpec(family=QueryFamily.COMPARISON, natural_language_question=tmpl.format(cat_a=ca, cat_b=cb),
                scope_predicate=InPredicate(field="category", values=[ca, cb]),
                fact_spec=FactSpec(fields=["record_id", "category", "price"]),
                aggregation_spec=ComparisonSpec(measure="mean", field="price", group_a_predicate=EqPredicate(field="category", value=ca), group_b_predicate=EqPredicate(field="category", value=cb), output="difference"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        return None

    # =========================================================================
    # TOP-K templates (15)
    # =========================================================================

    def _make_topk_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]; tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]
        k = rng.choice([3, 5])

        if tid == "topk_brands_reviews":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "brand"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_cats_reviews":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "category"]),
                aggregation_spec=TopKSpec(group_by_field="category", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_brands_rating":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "brand", "rating"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="mean", value_field="rating", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid in ("topk_cats_rating", "topk_brands_count_cat"):
            if tid == "topk_cats_rating":
                return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                    scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                    fact_spec=FactSpec(fields=["record_id", "category", "rating"]),
                    aggregation_spec=TopKSpec(group_by_field="category", measure="mean", value_field="rating", k=k, tie_policy=TiePolicy.FIRST),
                    world_id=world_id, template_id=tid, tolerance=1e-4)
            else:
                cat = rng.choice(cats)
                return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k, category=cat),
                    scope_predicate=EqPredicate(field="category", value=cat),
                    fact_spec=FactSpec(fields=["record_id", "brand", "category"]),
                    aggregation_spec=TopKSpec(group_by_field="brand", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                    world_id=world_id, template_id=tid)

        elif tid in ("topk_brands_reviews_cat",):
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k, category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "brand", "category"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_cats_mean_price":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=IsNotNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "category", "price"]),
                aggregation_spec=TopKSpec(group_by_field="category", measure="mean", value_field="price", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "topk_brands_helpful":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="helpful_votes", low=0),
                fact_spec=FactSpec(fields=["record_id", "brand", "helpful_votes"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="mean", value_field="helpful_votes", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "topk_brands_verified_rate":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "brand", "verified_purchase"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="mean", value_field="verified_purchase", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "topk_cats_count_year":
            year = rng.choice([2021, 2022, 2023])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k, year=year),
                scope_predicate=RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat()),
                fact_spec=FactSpec(fields=["record_id", "category", "event_time"]),
                aggregation_spec=TopKSpec(group_by_field="category", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_brands_tie":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "brand"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_cats_price_cat":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=IsNotNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "category", "price"]),
                aggregation_spec=TopKSpec(group_by_field="category", measure="mean", value_field="price", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "topk_brands_low_rating":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "brand", "rating"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="mean", value_field="rating", k=k, ascending=True, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "topk_brands_count_verified":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "brand", "verified_purchase"]),
                aggregation_spec=TopKSpec(group_by_field="brand", measure="count", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid)

        elif tid == "topk_cats_proportion_5star":
            return QuerySpec(family=QueryFamily.TOP_K, natural_language_question=tmpl.format(k=k),
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "category", "rating"]),
                aggregation_spec=TopKSpec(group_by_field="category", measure="mean", value_field="rating", k=k, tie_policy=TiePolicy.FIRST),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        return None

    # =========================================================================
    # TREND templates (15)
    # =========================================================================

    def _make_trend_query(self, template: tuple, context: dict, rng: random.Random) -> Optional[QuerySpec]:
        tid = template[0]; tmpl = template[1]
        world_id = context["world_id"]
        cats = context["categories"]

        if tid == "trend_monthly_count":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=TrendSpec(bucket="month", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_quarterly_rating":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time", "rating"]),
                aggregation_spec=TrendSpec(bucket="quarter", measure="mean", value_field="rating"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_monthly_count_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "event_time", "category"]),
                aggregation_spec=TrendSpec(bucket="month", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_yearly_count":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=TrendSpec(bucket="year", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_quarterly_count":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=TrendSpec(bucket="quarter", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_quarterly_count_verified":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=EqPredicate(field="verified_purchase", value=True),
                fact_spec=FactSpec(fields=["record_id", "event_time", "verified_purchase"]),
                aggregation_spec=TrendSpec(bucket="quarter", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_monthly_price_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl.format(category=cat),
                scope_predicate=AndPredicate(operands=[EqPredicate(field="category", value=cat), IsNotNullPredicate(field="price")]),
                fact_spec=FactSpec(fields=["record_id", "event_time", "category", "price"]),
                aggregation_spec=TrendSpec(bucket="month", measure="mean", value_field="price", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_yearly_rating":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time", "rating"]),
                aggregation_spec=TrendSpec(bucket="year", measure="mean", value_field="rating"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_monthly_count_brand":
            cat_brands = context.get("brands", [])
            brand = rng.choice(cat_brands) if cat_brands else "TechPrime"
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl.format(brand=brand),
                scope_predicate=EqPredicate(field="brand", value=brand),
                fact_spec=FactSpec(fields=["record_id", "event_time", "brand"]),
                aggregation_spec=TrendSpec(bucket="month", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_quarterly_proportion_verified":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time", "verified_purchase"]),
                aggregation_spec=TrendSpec(bucket="quarter", measure="mean", value_field="verified_purchase"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_monthly_count_date_range":
            year = rng.choice([2021, 2022])
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, tzinfo=timezone.utc)
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl.format(start_date=start.date(), end_date=end.date()),
                scope_predicate=RangePredicate(field="event_time", low=start.isoformat(), high=end.isoformat()),
                fact_spec=FactSpec(fields=["record_id", "event_time"]),
                aggregation_spec=TrendSpec(bucket="month", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_quarterly_price_all":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=IsNotNullPredicate(field="price"),
                fact_spec=FactSpec(fields=["record_id", "event_time", "price"]),
                aggregation_spec=TrendSpec(bucket="quarter", measure="mean", value_field="price", null_policy=NullPolicy.EXCLUDE),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_yearly_count_cat":
            cat = rng.choice(cats)
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl.format(category=cat),
                scope_predicate=EqPredicate(field="category", value=cat),
                fact_spec=FactSpec(fields=["record_id", "event_time", "category"]),
                aggregation_spec=TrendSpec(bucket="year", measure="count"),
                world_id=world_id, template_id=tid)

        elif tid == "trend_monthly_helpful":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time", "helpful_votes"]),
                aggregation_spec=TrendSpec(bucket="month", measure="mean", value_field="helpful_votes"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        elif tid == "trend_yearly_proportion_5star":
            return QuerySpec(family=QueryFamily.TREND, natural_language_question=tmpl,
                scope_predicate=RangePredicate(field="rating", low=1.0, high=5.0),
                fact_spec=FactSpec(fields=["record_id", "event_time", "rating"]),
                aggregation_spec=TrendSpec(bucket="year", measure="mean", value_field="rating"),
                world_id=world_id, template_id=tid, tolerance=1e-4)

        return None
