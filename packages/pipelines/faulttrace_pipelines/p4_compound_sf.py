"""
P4-COMPOUND-SF pipeline.

Fault model: Combines scope fault (P1) AND facts fault (P2).
Aggregation A is correct (oracle applied to the doubly-corrupted data).

This tests the interaction between scope and fact errors:
  fault(R) ∩ fault(E) — can the attribution engine decompose these?

The pipeline re-uses the perturbation functions from P1 and P2.
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
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.p1_wrong_scope import _perturb_predicate
from faulttrace_pipelines.p2_wrong_facts import _corrupt_dataframe
from faulttrace_core.predicates import compiler

PIPELINE_ID = "P4-compound-scope-facts"
PROVIDER_ID = "fault-injection"

_eval = PandasEvaluator()


class P4CompoundSF(AbstractPipeline):
    """
    P4 — Compound Scope + Facts fault injection pipeline.

    Fault: Both scope predicate and extracted facts are corrupted.
    Aggregation is correct (applied to doubly-corrupted data).
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
        rng = random.Random(str(query.query_id))

        events.append(self._make_event(
            run_id, "query_load", TraceEventType.QUERY_LOAD,
            f"P4 loaded query {query.query_id}",
            payload={"pipeline": self.pipeline_id, "fault": "compound_scope_facts"},
        ))

        # ── Stage 2: scope (FAULTY — P1 perturbation) ──
        t1 = time.perf_counter()
        wrong_pred = _perturb_predicate(query.scope_predicate, rng)
        try:
            mask = compiler.to_pandas_mask(wrong_pred, df)
            scope_df = df[mask].copy()
        except Exception:
            mask = compiler.to_pandas_mask(query.scope_predicate, df)
            scope_df = df[mask].copy()

        scope_duration = (time.perf_counter() - t1) * 1000
        events.append(self._make_event(
            run_id, "scope_enumerate", TraceEventType.SCOPE_ENUMERATE,
            f"P4 FAULT scope: correct={len(df[compiler.to_pandas_mask(query.scope_predicate, df)])} "
            f"wrong={len(scope_df)}",
            record_count_in=len(df),
            record_count_out=len(scope_df),
            duration_ms=scope_duration,
            payload={"fault_type": "wrong_scope"},
        ))

        scope_path = self.artifacts_dir / run_id / "scope_output.parquet"
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        scope_df.to_parquet(scope_path, index=False)
        
        # ── Stage 3: facts (FAULTY — P2 corruption) ──
        t2 = time.perf_counter()
        fields = query.fact_spec.fields
        avail = [f for f in fields if f in scope_df.columns]
        raw_extraction = scope_df[avail].copy() if avail else scope_df.copy()
        corrupted_df = _corrupt_dataframe(raw_extraction, avail, rng)
        extract_duration = (time.perf_counter() - t2) * 1000

        events.append(self._make_event(
            run_id, "fact_extract", TraceEventType.FACT_EXTRACT,
            f"P4 FAULT facts: noise injected into {len(avail)} fields",
            record_count_in=len(scope_df),
            record_count_out=len(corrupted_df),
            duration_ms=extract_duration,
            payload={"fault_type": "wrong_facts"},
        ))

        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        corrupted_df.to_parquet(extract_path, index=False)
        
        # ── Stage 4: aggregate (correct on doubly-corrupted data) ──
        t3 = time.perf_counter()
        eval_df = df.copy()
        for col in corrupted_df.columns:
            if col in eval_df.columns and col not in ("record_id", "world_id"):
                eval_df.loc[corrupted_df.index, col] = corrupted_df[col].values

        wrong_query = query.model_copy(update={"scope_predicate": wrong_pred})
        agg_result = _eval.evaluate(wrong_query, eval_df)
        answer_value = agg_result.get("result")
        agg_duration = (time.perf_counter() - t3) * 1000

        events.append(self._make_event(
            run_id, "aggregate", TraceEventType.AGGREGATE,
            f"P4 aggregate on doubly-corrupted data → {answer_value}",
            duration_ms=agg_duration,
            payload={"answer": str(answer_value), "fault_layer": "scope+facts"},
        ))

        events.append(self._make_event(
            run_id, "validate", TraceEventType.VALIDATE,
            f"Gold: {gold_answer.answer_value if gold_answer else '?'} vs pipeline: {answer_value}",
        ))
        events.append(self._make_event(
            run_id, "persist", TraceEventType.PERSIST, "P4 artifacts persisted",
        ))

        return answer_value, events, components, 0, 0
