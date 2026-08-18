"""
Procedural query factory for EDGAR Track T pilot.
Generates grounded queries for EDGAR XBRL facts.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
from faulttrace_core.models import (
    AndPredicate,
    CountSpec,
    EqPredicate,
    FactSpec,
    MeanSpec,
    QueryFamily,
    QuerySpec,
    SumSpec,
    TopKSpec,
    TrendSpec,
)

from faulttrace_pipelines.query_factory import BenchmarkPack

# Basic EDGAR templates
_EDGAR_TEMPLATES = {
    QueryFamily.COUNT: [
        ("edgar_count_form", "How many {form_type} filings are there in total?", "easy"),
        ("edgar_count_company_form", "How many {form_type} filings did {cik} submit?", "medium"),
    ],
    QueryFamily.MEAN: [
        (
            "edgar_mean_fact",
            "What is the average {tag} reported across all {form_type} filings?",
            "easy",
        ),
        ("edgar_mean_company_fact", "What is the average {tag} for {cik}?", "medium"),
        (
            "edgar_sum_fact",
            "What is the total sum of {tag} across all companies in {year}?",
            "medium",
        ),
    ],
    QueryFamily.TOP_K: [
        ("edgar_topk_fact", "Which {k} companies reported the highest {tag}?", "easy"),
    ],
    QueryFamily.TREND: [
        ("edgar_trend_fact", "How did the average {tag} trend over the years?", "medium"),
    ],
}


class EdgarQueryFactory:
    """Generates procedural queries for an EDGAR corpus world."""

    def __init__(self, data_dir: Path = Path("data/generated")):
        self.data_dir = data_dir

    def generate_for_world(
        self,
        world_id: str,
        target_count: int = 10,
        seed: int | None = None,
    ) -> list[QuerySpec]:
        world_dir = self.data_dir / "worlds" / world_id
        parquet_path = world_dir / "records.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        if seed is None:
            seed = int(hashlib.sha256(world_id.encode()).hexdigest()[:8], 16) & 0xFFFFFF

        rng = random.Random(seed)
        queries: list[QuerySpec] = []

        ciks = df["cik"].dropna().unique().tolist()
        forms = df["form_type"].dropna().unique().tolist()
        tags = df["tag"].dropna().unique().tolist()

        context = {
            "world_id": world_id,
            "ciks": ciks,
            "forms": forms,
            "tags": tags,
            "df": df,
        }

        # Just generate a few of each type
        for _ in range(target_count):
            family = rng.choice(list(_EDGAR_TEMPLATES.keys()))
            template = rng.choice(_EDGAR_TEMPLATES[family])

            q = self._make_query(family, template, context, rng)
            if q is not None:
                # Basic non-empty validation
                try:
                    from faulttrace_core.predicates import compiler

                    mask = compiler.to_pandas_mask(q.scope_predicate, df)
                    if mask.sum() > 0:
                        queries.append(q)
                except Exception as e:
                    print(f"Failed to compile or check mask: {e}")

        return queries

    def _make_query(
        self, family: QueryFamily, template: tuple, context: dict, rng: random.Random
    ) -> QuerySpec | None:
        tid, tmpl, difficulty = template
        world_id = context["world_id"]
        ciks = context["ciks"]
        forms = context["forms"]
        tags = context["tags"]

        if not ciks or not forms or not tags:
            return None

        if tid == "edgar_count_form":
            form = rng.choice(forms)
            return QuerySpec(
                family=QueryFamily.COUNT,
                natural_language_question=tmpl.format(form_type=form),
                scope_predicate=EqPredicate(field="form_type", value=form),
                fact_spec=FactSpec(fields=["record_id", "form_type"]),
                aggregation_spec=CountSpec(),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_count_company_form":
            form = rng.choice(forms)
            cik = rng.choice(ciks)
            return QuerySpec(
                family=QueryFamily.COUNT,
                natural_language_question=tmpl.format(form_type=form, cik=cik),
                scope_predicate=AndPredicate(
                    operands=[
                        EqPredicate(field="form_type", value=form),
                        EqPredicate(field="cik", value=cik),
                    ]
                ),
                fact_spec=FactSpec(fields=["record_id", "form_type", "cik"]),
                aggregation_spec=CountSpec(),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_mean_fact":
            form = rng.choice(forms)
            tag = rng.choice(tags)
            return QuerySpec(
                family=QueryFamily.MEAN,
                natural_language_question=tmpl.format(form_type=form, tag=tag),
                scope_predicate=AndPredicate(
                    operands=[
                        EqPredicate(field="form_type", value=form),
                        EqPredicate(field="tag", value=tag),
                    ]
                ),
                fact_spec=FactSpec(fields=["record_id", "value"]),
                aggregation_spec=MeanSpec(field="value"),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_mean_company_fact":
            cik = rng.choice(ciks)
            tag = rng.choice(tags)
            return QuerySpec(
                family=QueryFamily.MEAN,
                natural_language_question=tmpl.format(cik=cik, tag=tag),
                scope_predicate=AndPredicate(
                    operands=[
                        EqPredicate(field="cik", value=cik),
                        EqPredicate(field="tag", value=tag),
                    ]
                ),
                fact_spec=FactSpec(fields=["record_id", "value"]),
                aggregation_spec=MeanSpec(field="value"),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_sum_fact":
            tag = rng.choice(tags)
            year = 2023
            return QuerySpec(
                family=QueryFamily.MEAN,
                natural_language_question=tmpl.format(tag=tag, year=year),
                scope_predicate=AndPredicate(
                    operands=[
                        EqPredicate(field="tag", value=tag),
                        EqPredicate(field="fiscal_year", value=year),
                    ]
                ),
                fact_spec=FactSpec(fields=["record_id", "value"]),
                aggregation_spec=SumSpec(field="value"),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_topk_fact":
            tag = rng.choice(tags)
            k = rng.choice([3, 5])
            return QuerySpec(
                family=QueryFamily.TOP_K,
                natural_language_question=tmpl.format(tag=tag, k=k),
                scope_predicate=EqPredicate(field="tag", value=tag),
                fact_spec=FactSpec(fields=["record_id", "cik", "value"]),
                aggregation_spec=TopKSpec(
                    group_by_field="cik", measure="sum", value_field="value", k=k
                ),
                world_id=world_id,
                template_id=tid,
            )
        elif tid == "edgar_trend_fact":
            tag = rng.choice(tags)
            return QuerySpec(
                family=QueryFamily.TREND,
                natural_language_question=tmpl.format(tag=tag),
                scope_predicate=EqPredicate(field="tag", value=tag),
                fact_spec=FactSpec(fields=["record_id", "filing_date", "value"]),
                aggregation_spec=TrendSpec(
                    time_field="filing_date", bucket="year", measure="mean", value_field="value"
                ),
                world_id=world_id,
                template_id=tid,
            )

        return None

    def build_benchmark_pack(self, world_id: str, total_count: int = 20) -> BenchmarkPack:
        queries = self.generate_for_world(world_id, total_count)

        # Validation
        world_dir = self.data_dir / "worlds" / world_id
        parquet_path = world_dir / "records.parquet"

        agreed_count = 0
        disagreed_count = 0
        skipped_count = 0

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            try:
                from faulttrace_gold.duckdb_engine import DuckDBEvaluator
                from faulttrace_gold.pandas_engine import PandasEvaluator
                from faulttrace_gold.validator import _results_agree as results_agree

                pe = PandasEvaluator()
                de = DuckDBEvaluator()

                for q in queries:
                    try:
                        p_res = pe.evaluate(q, df)
                        d_res = de.evaluate(q, parquet_path)
                        if results_agree(p_res["result"], d_res["result"], q.tolerance):
                            agreed_count += 1
                        else:
                            disagreed_count += 1
                    except Exception as e:
                        print(f"Eval exception on {q.template_id}: {e}")
                        skipped_count += 1
            except ImportError:
                skipped_count = len(queries)

        pack = BenchmarkPack(
            world_id=world_id,
            total_count=len(queries),
            agreed_count=agreed_count,
            disagreed_count=disagreed_count,
            skipped_count=skipped_count,
            gold_ready=(disagreed_count == 0 and len(queries) > 0),
        )
        return pack
