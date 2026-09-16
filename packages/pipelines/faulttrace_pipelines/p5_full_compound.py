"""
P5-FULL-COMPOUND pipeline.

Fault model: All three components are faulty: scope (R), facts (E), aggregation (A).

This is the hardest case to attribute. The system must decompose the total error
into three additive components:
    total_error ≈ error(R) + error(E) + error(A) + interaction(R,E,A)

The attribution engine (counterfactual engine) does oracle replacement one
component at a time:
    REF(R) = |gold - run_with_correct_R| / |gold|
    REF(E) = |gold - run_with_correct_E| / |gold|
    REF(A) = |gold - run_with_correct_A| / |gold|
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from faulttrace_core.models import (
    GoldAnswer,
    QuerySpec,
    TraceEventType,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator

from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.p1_wrong_scope import _perturb_predicate
from faulttrace_pipelines.p2_wrong_facts import _corrupt_dataframe, _overlay_corrupted_fields
from faulttrace_pipelines.p3_wrong_aggregation import _corrupt_answer

PIPELINE_ID = "P5-full-compound"
PROVIDER_ID = "fault-injection"

_eval = PandasEvaluator()


class P5FullCompound(AbstractPipeline):
    """
    P5 — All-faults compound pipeline.

    Fault: scope, facts, AND aggregation are all corrupted.
    This pipeline exercises the full attribution decomposition.
    """

    pipeline_id = PIPELINE_ID
    provider_id = PROVIDER_ID

    def replay_extraction(self, query: QuerySpec, scoped_df: pd.DataFrame) -> list[dict[str, Any]]:
        rng = self.fault_rng(query, "facts")
        fields = query.fact_spec.fields
        fact_df = _eval.extract_facts(query.fact_spec, scoped_df)
        return _corrupt_dataframe(fact_df, fields, rng).to_dict(orient="records")

    def replay_aggregation(self, query: QuerySpec, extracted_rows: list[dict[str, Any]]) -> Any:
        correct_answer = super().replay_aggregation(query, extracted_rows)
        return _corrupt_answer(correct_answer, query, self.fault_rng(query, "aggregation"))

    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Path | None,
        gold_answer: GoldAnswer | None,
    ) -> tuple[Any, list, list, int, int]:
        events: list = []
        components: list = []
        scope_rng = self.fault_rng(query, "scope")
        facts_rng = self.fault_rng(query, "facts")
        aggregation_rng = self.fault_rng(query, "aggregation")

        events.append(
            self._make_event(
                run_id,
                "query_load",
                TraceEventType.QUERY_LOAD,
                f"P5 loaded query {query.query_id}",
                payload={"pipeline": self.pipeline_id, "fault": "full_compound_R+E+A"},
            )
        )

        # ── Stage 2: scope (FAULTY) ──
        t1 = time.perf_counter()
        wrong_pred = _perturb_predicate(query.scope_predicate, scope_rng)
        try:
            mask = compiler.to_pandas_mask(wrong_pred, df)
            scope_df = df[mask].copy()
        except Exception:
            mask = compiler.to_pandas_mask(query.scope_predicate, df)
            scope_df = df[mask].copy()

        events.append(
            self._make_event(
                run_id,
                "scope_enumerate",
                TraceEventType.SCOPE_ENUMERATE,
                f"P5 FAULT scope: {len(scope_df)} rows (faulted)",
                record_count_in=len(df),
                record_count_out=len(scope_df),
                duration_ms=(time.perf_counter() - t1) * 1000,
                payload={"fault_type": "wrong_scope"},
            )
        )

        scope_path = self.artifacts_dir / run_id / "scope_output.parquet"
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        scope_df.to_parquet(scope_path, index=False)

        # ── Stage 3: facts (FAULTY) ──
        t2 = time.perf_counter()
        fields = query.fact_spec.fields
        avail = [f for f in fields if f in scope_df.columns]
        lineage_fields = (["record_id"] if "record_id" in scope_df.columns else []) + avail
        raw_extraction = scope_df[list(dict.fromkeys(lineage_fields))].copy()
        corrupted_df = _corrupt_dataframe(raw_extraction, avail, facts_rng)

        events.append(
            self._make_event(
                run_id,
                "fact_extract",
                TraceEventType.FACT_EXTRACT,
                f"P5 FAULT facts: {len(avail)} fields corrupted",
                record_count_in=len(scope_df),
                record_count_out=len(corrupted_df),
                duration_ms=(time.perf_counter() - t2) * 1000,
                payload={"fault_type": "wrong_facts"},
            )
        )

        extract_path = self.artifacts_dir / run_id / "extraction.parquet"
        corrupted_df.to_parquet(extract_path, index=False)

        # ── Stage 4: aggregate (FAULTY — compute then corrupt) ──
        t3 = time.perf_counter()
        eval_df = _overlay_corrupted_fields(df, corrupted_df)

        wrong_query = query.model_copy(update={"scope_predicate": wrong_pred})
        intermediate = _eval.evaluate(wrong_query, eval_df)
        intermediate_answer = intermediate.get("result")

        # Now corrupt the aggregation output too
        answer_value = _corrupt_answer(intermediate_answer, query, aggregation_rng)

        events.append(
            self._make_event(
                run_id,
                "aggregate",
                TraceEventType.AGGREGATE,
                f"P5 FAULT aggregate: intermediate={intermediate_answer} → final={answer_value}",
                duration_ms=(time.perf_counter() - t3) * 1000,
                payload={
                    "fault_type": "wrong_aggregation",
                    "intermediate": str(intermediate_answer),
                    "final": str(answer_value),
                    "fault_layer": "R+E+A",
                },
            )
        )

        events.append(
            self._make_event(
                run_id,
                "validate",
                TraceEventType.VALIDATE,
                f"Gold: {gold_answer.answer_value if gold_answer else '?'} vs pipeline: {answer_value}",
            )
        )
        events.append(
            self._make_event(
                run_id,
                "persist",
                TraceEventType.PERSIST,
                "P5 artifacts persisted",
            )
        )

        return answer_value, events, components, 0, 0
