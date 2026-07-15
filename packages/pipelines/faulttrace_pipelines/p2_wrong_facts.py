"""
P2-WRONG-FACTS pipeline.

Fault model: The pipeline applies the correct scope R but extracts *incorrect*
structured facts E. Specifically it introduces field-level noise:
  - numeric fields: Gaussian noise scaled to 10% of the field range
  - string fields: random replacement from a pool of plausible values
  - boolean fields: random bit-flip with 30% probability

This simulates the failure mode where fact extraction (e.g., NLP entity
extraction from review text) produces corrupted structured facts.

Aggregation A is then applied correctly to the corrupted facts,
so the fault is purely in E: R and A are both correct.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from faulttrace_core.models import (
    ComponentOutput,
    GoldAnswer,
    QuerySpec,
    TraceEventType,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline

PIPELINE_ID = "P2-wrong-facts"
PROVIDER_ID = "fault-injection"

_eval = PandasEvaluator()

_CATEGORY_POOL = [
    "Electronics", "Books", "Sports", "Clothing", "Home & Kitchen",
    "Automotive", "Health", "Office Products", "Toys", "Garden",
]
_BRAND_POOL = ["TechPrime", "VoltEdge", "BookCo", "SportPeak", "FitCore", "OfficePro", "HomePlus"]


def _corrupt_dataframe(df: pd.DataFrame, fields: list[str], rng: random.Random) -> pd.DataFrame:
    """Apply deterministic noise to the extracted fields."""
    df = df.copy()

    for field in fields:
        if field not in df.columns:
            continue

        col = df[field]
        dtype = col.dtype

        if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
            # 10% Gaussian noise on numeric fields
            clean = col.dropna()
            if len(clean) == 0:
                continue
            field_range = float(clean.max() - clean.min()) or 1.0
            noise = pd.Series(
                [rng.gauss(0, 0.10 * field_range) for _ in range(len(df))],
                index=df.index,
            )
            df[field] = (col + noise).clip(lower=0)
            # Round rating-like fields
            if field == "rating":
                df[field] = df[field].clip(1.0, 5.0).round(1)

        elif pd.api.types.is_bool_dtype(dtype) or col.isin([True, False]).all():
            # 30% bit-flip on boolean fields
            df[field] = col.apply(
                lambda v: (not v) if rng.random() < 0.30 else v
            )

        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # Replace 20% of string values with plausible wrong value
            if field == "category":
                df[field] = col.apply(
                    lambda v: rng.choice([c for c in _CATEGORY_POOL if c != v]) if rng.random() < 0.20 else v
                )
            elif field == "brand":
                df[field] = col.apply(
                    lambda v: rng.choice([b for b in _BRAND_POOL if b != v]) if rng.random() < 0.20 else v
                )

    return df


class P2WrongFacts(AbstractPipeline):
    """
    P2 — Wrong Fact Extraction fault injection pipeline.

    Fault: corrupt extracted field values with Gaussian / replacement noise.
    Scope and aggregation are both correct (oracle).
    """

    pipeline_id = PIPELINE_ID
    provider_id = PROVIDER_ID

    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path],
        gold_answer: Optional[GoldAnswer],
    ) -> tuple[Any, list, list, int, int]:
        events: list = []
        components: list = []

        # ── Stage 1: query_load ──
        events.append(self._make_event(
            run_id, "query_load", TraceEventType.QUERY_LOAD,
            f"P2 loaded query {query.query_id} family={query.family.value}",
            payload={"pipeline": self.pipeline_id, "fault": "wrong_facts"},
        ))

        # ── Stage 2: scope_enumerate (correct) ──
        t1 = time.perf_counter()
        mask = compiler.to_pandas_mask(query.scope_predicate, df)
        scope_df = df[mask].copy()
        scope_duration = (time.perf_counter() - t1) * 1000

        events.append(self._make_event(
            run_id, "scope_enumerate", TraceEventType.SCOPE_ENUMERATE,
            f"Correct scope: {len(scope_df)} records",
            record_count_in=len(df),
            record_count_out=len(scope_df),
            duration_ms=scope_duration,
        ))

        scope_path = self.artifacts_dir / run_id / "scope_output.parquet"
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        scope_df.to_parquet(scope_path, index=False)
        
        # ── Stage 3: fact_extract (FAULTY — adds noise) ──
        t2 = time.perf_counter()
        rng = random.Random(str(query.query_id))
        fields = query.fact_spec.fields
        avail = [f for f in fields if f in scope_df.columns]
        raw_extraction = scope_df[avail].copy() if avail else scope_df.copy()
        corrupted_df = _corrupt_dataframe(raw_extraction, avail, rng)
        extract_duration = (time.perf_counter() - t2) * 1000

        events.append(self._make_event(
            run_id, "fact_extract", TraceEventType.FACT_EXTRACT,
            f"P2 FAULT: field noise injected into {len(avail)} fields across {len(corrupted_df)} records",
            record_count_in=len(scope_df),
            record_count_out=len(corrupted_df),
            duration_ms=extract_duration,
            payload={
                "fault_type": "wrong_facts",
                "corrupted_fields": avail,
                "noise_model": "gaussian_10pct + categorical_20pct_swap",
            },
        ))

        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        corrupted_df.to_parquet(extract_path, index=False)
        
        # ── Stage 4: aggregate (correct oracle on corrupted data) ──
        t3 = time.perf_counter()
        # Re-join corrupted fields back into full df for evaluation
        eval_df = df.copy()
        for col in corrupted_df.columns:
            if col in eval_df.columns and col not in ("record_id", "world_id"):
                eval_df.loc[corrupted_df.index, col] = corrupted_df[col].values

        agg_result = _eval.evaluate(query, eval_df)
        answer_value = agg_result.get("result")
        agg_duration = (time.perf_counter() - t3) * 1000

        events.append(self._make_event(
            run_id, "aggregate", TraceEventType.AGGREGATE,
            f"Aggregation on corrupted facts → {answer_value}",
            duration_ms=agg_duration,
            payload={"answer": str(answer_value), "fault_layer": "facts"},
        ))

        events.append(self._make_event(
            run_id, "validate", TraceEventType.VALIDATE,
            f"Gold: {gold_answer.answer_value if gold_answer else '?'} vs pipeline: {answer_value}",
        ))
        events.append(self._make_event(
            run_id, "persist", TraceEventType.PERSIST,
            "P2 run artifacts persisted",
        ))

        return answer_value, events, components, 0, 0
