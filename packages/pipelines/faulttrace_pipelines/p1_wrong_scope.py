"""
P1-WRONG-SCOPE pipeline.

Fault model: The pipeline uses an incorrect evidence scope — it applies a
*randomly-perturbed* predicate that widens or narrows the actual scope R.
This simulates the failure mode where the retrieval step retrieves wrong records.

Perturbation strategy (deterministic given the query_id seed):
  - COUNT/MEAN/PROPORTION: replace a numeric threshold with ±20% jitter
  - TOP_K: inject an extra EqPredicate on a random non-scope field
  - COMPARISON/TREND: randomly flip the scope to a different category value

The pipeline then runs the deterministic gold aggregation on the WRONG scope,
so the fault is purely in R: E and A are both correct.

Recoverable-error attribution: REF(R) = |gold - pipeline_answer| / |gold|
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import time

import pandas as pd

from faulttrace_core.models import (
    AndPredicate,
    ComponentOutput,
    EqPredicate,
    GoldAnswer,
    InPredicate,
    IsNotNullPredicate,
    IsNullPredicate,
    QuerySpec,
    RangePredicate,
    TraceEventType,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline

PIPELINE_ID = "P1-wrong-scope"
PROVIDER_ID = "fault-injection"

_eval = PandasEvaluator()

# Categories available for scope perturbation
_CATEGORIES = [
    "Electronics", "Books", "Sports", "Clothing", "Home & Kitchen",
    "Automotive", "Health", "Office Products", "Toys", "Garden",
]
_BRANDS = ["TechPrime", "VoltEdge", "BookCo", "SportPeak", "FitCore", "OfficePro"]


def _perturb_predicate(pred: Any, rng: random.Random) -> Any:
    """Apply a deterministic fault to the scope predicate."""
    if isinstance(pred, RangePredicate):
        # Widen or narrow the range by ±20%
        jitter = rng.uniform(0.8, 1.2)
        new_low = pred.low
        new_high = pred.high
        if pred.low is not None:
            try:
                new_low = float(pred.low) * jitter
            except (TypeError, ValueError):
                pass
        if pred.high is not None:
            try:
                new_high = float(pred.high) * (2.0 - jitter)
            except (TypeError, ValueError):
                pass
        return RangePredicate(
            field=pred.field,
            low=new_low,
            high=new_high,
            low_inclusive=pred.low_inclusive,
            high_inclusive=pred.high_inclusive,
        )

    elif isinstance(pred, EqPredicate):
        # Swap to a different category/brand value
        if pred.field == "category":
            wrong = rng.choice([c for c in _CATEGORIES if c != pred.value])
            return EqPredicate(field="category", value=wrong)
        elif pred.field == "brand":
            wrong = rng.choice([b for b in _BRANDS if b != pred.value])
            return EqPredicate(field="brand", value=wrong)
        elif pred.field in ("rating",):
            # Shift rating threshold
            try:
                shifted = max(1.0, min(5.0, float(pred.value) + rng.uniform(-1.0, 1.0)))
                return EqPredicate(field=pred.field, value=round(shifted, 1))
            except (TypeError, ValueError):
                return pred

    elif isinstance(pred, AndPredicate):
        # Perturb the first operand only
        operands = list(pred.operands)
        if operands:
            operands[0] = _perturb_predicate(operands[0], rng)
        return AndPredicate(operands=operands)

    elif isinstance(pred, InPredicate):
        # Remove one value from the in-list (narrows scope)
        values = list(pred.values)
        if len(values) > 1:
            values.pop(rng.randint(0, len(values) - 1))
        return InPredicate(field=pred.field, values=values)

    # Fallback: add an always-true range to slightly change evaluation path
    return RangePredicate(field="rating", low=rng.uniform(1.0, 2.0), high=5.0)


class P1WrongScope(AbstractPipeline):
    """
    P1 — Wrong Evidence Scope fault injection pipeline.

    Fault: perturbs the scope predicate deterministically (seeded from query_id).
    Fact extraction and aggregation are both correct (oracle).
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
        t0 = time.perf_counter()
        events.append(self._make_event(
            run_id, "query_load", TraceEventType.QUERY_LOAD,
            f"P1 loaded query {query.query_id} family={query.family.value}",
            payload={"pipeline": self.pipeline_id, "fault": "wrong_scope"},
        ))

        # ── Stage 2: scope_enumerate (FAULTY) ──
        t1 = time.perf_counter()
        rng = random.Random(str(query.query_id))
        wrong_pred = _perturb_predicate(query.scope_predicate, rng)

        # Apply wrong predicate
        try:
            mask = compiler.to_pandas_mask(wrong_pred, df)
            scope_df = df[mask].copy()
        except Exception:
            # Fall back to original scope if perturbed predicate errors
            mask = compiler.to_pandas_mask(query.scope_predicate, df)
            scope_df = df[mask].copy()

        scope_duration = (time.perf_counter() - t1) * 1000
        events.append(self._make_event(
            run_id, "scope_enumerate", TraceEventType.SCOPE_ENUMERATE,
            f"P1 FAULT: wrong scope predicate applied. "
            f"Correct scope would have {len(df[compiler.to_pandas_mask(query.scope_predicate, df)])} rows; "
            f"wrong scope has {len(scope_df)} rows.",
            record_count_in=len(df),
            record_count_out=len(scope_df),
            duration_ms=scope_duration,
            payload={
                "fault_type": "wrong_scope",
                "correct_rows": int(len(df[compiler.to_pandas_mask(query.scope_predicate, df)])),
                "wrong_rows": int(len(scope_df)),
                "predicate_type": type(wrong_pred).__name__,
            },
        ))

        # Save scope artifact
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
            f"Extracted {len(avail)} fields from {len(extraction_df)} records",
            record_count_in=len(scope_df),
            record_count_out=len(extraction_df),
            duration_ms=extract_duration,
        ))

        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        extraction_df.to_parquet(extract_path, index=False)
        
        # ── Stage 4: aggregate (correct oracle on wrong data) ──
        t3 = time.perf_counter()
        # Build a modified query with the wrong scope but same aggregation spec
        wrong_query = query.model_copy(update={"scope_predicate": wrong_pred})
        agg_result = _eval.evaluate(wrong_query, df)
        answer_value = agg_result.get("result")
        agg_duration = (time.perf_counter() - t3) * 1000

        events.append(self._make_event(
            run_id, "aggregate", TraceEventType.AGGREGATE,
            f"Aggregation on wrong scope → {answer_value}",
            duration_ms=agg_duration,
            payload={"answer": str(answer_value), "fault_layer": "scope"},
        ))

        # ── Stage 5: validate ──
        events.append(self._make_event(
            run_id, "validate", TraceEventType.VALIDATE,
            f"Gold comparison: answer={answer_value} gold={gold_answer.answer_value if gold_answer else '?'}",
            payload={"is_scope_fault": True},
        ))

        # ── Stage 6: persist ──
        events.append(self._make_event(
            run_id, "persist", TraceEventType.PERSIST,
            "P1 run artifacts persisted",
        ))

        return answer_value, events, components, 0, 0
