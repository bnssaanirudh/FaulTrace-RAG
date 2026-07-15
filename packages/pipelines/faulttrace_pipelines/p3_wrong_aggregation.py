"""
P3-WRONG-AGGREGATION pipeline.

Fault model: The pipeline applies the correct scope R and correct facts E
but performs an *incorrect aggregation* A.

Perturbation strategy:
  - COUNT → returns count * random_factor in [0.7, 1.4]
  - MEAN → returns mean ± (0.5 * std_dev) with directional bias
  - PROPORTION → shifts proportion by ±0.15
  - COMPARISON → swaps the sign of the difference (wrong direction)
  - TOP_K → reverses the ranking (worst-first instead of best-first)
  - TREND → returns buckets in wrong order or drops one bucket

The fault is purely in A: R and E are both correct (oracle).
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from faulttrace_core.models import (
    ComparisonSpec,
    ComponentOutput,
    CountSpec,
    GoldAnswer,
    MeanSpec,
    ProportionSpec,
    QueryFamily,
    QuerySpec,
    TopKSpec,
    TrendSpec,
    TraceEventType,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline

PIPELINE_ID = "P3-wrong-aggregation"
PROVIDER_ID = "fault-injection"

_eval = PandasEvaluator()


def _corrupt_answer(answer: Any, query: QuerySpec, rng: random.Random) -> Any:
    """Apply deterministic corruption to the correct aggregation result."""
    spec = query.aggregation_spec
    family = query.family

    if answer is None:
        return answer

    if family == QueryFamily.COUNT:
        # Multiply by random factor in [0.7, 1.4]
        factor = rng.uniform(0.7, 1.4)
        return max(0, round(answer * factor))

    elif family == QueryFamily.MEAN:
        # Shift by ±0.5 std dev (simulate wrong computation)
        try:
            shift = rng.gauss(0, 0.5)
            return round(float(answer) + shift, 4)
        except (TypeError, ValueError):
            return answer

    elif family == QueryFamily.PROPORTION:
        try:
            shift = rng.uniform(-0.15, 0.15)
            corrupted = float(answer) + shift
            return round(max(0.0, min(1.0, corrupted)), 4)
        except (TypeError, ValueError):
            return answer

    elif family == QueryFamily.COMPARISON:
        # Flip the sign of the difference
        try:
            return round(-float(answer), 4)
        except (TypeError, ValueError):
            return answer

    elif family == QueryFamily.TOP_K:
        # Reverse the top-k ranking (worst becomes first)
        if isinstance(answer, list) and len(answer) > 1:
            return list(reversed(answer))
        return answer

    elif family == QueryFamily.TREND:
        # Shuffle the buckets (wrong temporal ordering)
        if isinstance(answer, list) and len(answer) > 1:
            shuffled = answer.copy()
            rng.shuffle(shuffled)
            return shuffled
        return answer

    return answer


class P3WrongAggregation(AbstractPipeline):
    """
    P3 — Wrong Aggregation fault injection pipeline.

    Fault: computes the correct intermediate result then applies a deterministic
    corruption to the final aggregation output.
    Scope and fact extraction are both correct (oracle).
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
            f"P3 loaded query {query.query_id} family={query.family.value}",
            payload={"pipeline": self.pipeline_id, "fault": "wrong_aggregation"},
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
        
        # ── Stage 3: fact_extract (correct) ──
        t2 = time.perf_counter()
        fields = query.fact_spec.fields
        avail = [f for f in fields if f in scope_df.columns]
        extraction_df = scope_df[avail].copy() if avail else scope_df.copy()
        extract_duration = (time.perf_counter() - t2) * 1000

        events.append(self._make_event(
            run_id, "fact_extract", TraceEventType.FACT_EXTRACT,
            f"Correct extraction: {len(avail)} fields from {len(extraction_df)} records",
            record_count_in=len(scope_df),
            record_count_out=len(extraction_df),
            duration_ms=extract_duration,
        ))

        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        extraction_df.to_parquet(extract_path, index=False)
        
        # ── Stage 4: aggregate (FAULTY — correct computation then corrupted) ──
        t3 = time.perf_counter()
        correct_result = _eval.evaluate(query, df)
        correct_answer = correct_result.get("result")

        rng = random.Random(str(query.query_id))
        answer_value = _corrupt_answer(correct_answer, query, rng)
        agg_duration = (time.perf_counter() - t3) * 1000

        events.append(self._make_event(
            run_id, "aggregate", TraceEventType.AGGREGATE,
            f"P3 FAULT: correct={correct_answer} → corrupted={answer_value}",
            duration_ms=agg_duration,
            payload={
                "fault_type": "wrong_aggregation",
                "correct_answer": str(correct_answer),
                "corrupted_answer": str(answer_value),
                "family": query.family.value,
            },
        ))

        events.append(self._make_event(
            run_id, "validate", TraceEventType.VALIDATE,
            f"Gold: {gold_answer.answer_value if gold_answer else '?'} vs pipeline: {answer_value}",
        ))
        events.append(self._make_event(
            run_id, "persist", TraceEventType.PERSIST,
            "P3 run artifacts persisted",
        ))

        return answer_value, events, components, 0, 0
