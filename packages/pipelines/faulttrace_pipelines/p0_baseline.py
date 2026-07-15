"""
P0-deterministic-scope-baseline pipeline.

Label: P0-DETERMINISTIC-SCOPE-BASELINE
This is NOT the final P0 model pipeline. It is a deterministic foundation baseline
that establishes the end-to-end run path before real LLM integrations are added in Prompt 2-3.

Pipeline steps:
1. Query load
2. Scope enumeration (safe predicate AST)
3. Fact extraction (field selection, no LLM)
4. Aggregation (deterministic reducer)
5. Validation (compare to gold)
6. Persistence (immutable artifact hashes)

Each step emits TraceEvents.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
import orjson

from faulttrace_core.models import (
    ComponentOutput,
    CoverageCertificate,
    GoldAnswer,
    PipelineRun,
    QuerySpec,
    RunStatus,
    TraceEvent,
    TraceEventType,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_pipelines.base import AbstractPipeline

PIPELINE_ID = "P0-deterministic-scope-baseline"
PROVIDER_ID = "deterministic"

pandas_eval = PandasEvaluator()


class P0DeterministicBaseline(AbstractPipeline):
    """
    Deterministic foundation baseline pipeline (P0).
    
    Uses the safe scope engine and deterministic aggregation.
    No LLM required. Emits complete TraceEvents for all stages.
    """
    
    pipeline_id = PIPELINE_ID
    provider_id = PROVIDER_ID

    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        super().__init__(artifacts_dir)

    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path],
        gold_answer: Optional[GoldAnswer],
    ) -> tuple[Any, list[TraceEvent], list[ComponentOutput], int, int]:
        # P0 overrides `run()` completely, so `_execute` is never called.
        return None, [], [], 0, 0

    def run(
        self,
        query: QuerySpec,
        df: pd.DataFrame,
        gold_answer: Optional[GoldAnswer] = None,
        parquet_path: Optional[Path] = None,
    ) -> tuple[PipelineRun, list[TraceEvent], list[ComponentOutput]]:
        """
        Execute the baseline pipeline.
        
        Returns (PipelineRun, [TraceEvent], [ComponentOutput]).
        """
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        
        # Create run directory
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Build run
        run = PipelineRun(
            run_id=run_id,
            query_id=query.query_id,
            pipeline_id=PIPELINE_ID,
            provider_id=PROVIDER_ID,
            started_at=started_at,
            status=RunStatus.RUNNING,
        )
        run = run.model_copy(update={"config_hash": run.compute_config_hash(query)})
        
        events: list[TraceEvent] = []
        components: list[ComponentOutput] = []
        answer = None
        error = None

        t0 = time.monotonic()

        try:
            # Stage 1: Query load
            ev1, t0 = self._stage_query_load(run_id, query, t0)
            events.append(ev1)

            # Stage 2: Scope enumeration
            ev2, scoped_df, t0 = self._stage_scope_enumerate(run_id, query, df, ev1.event_id, t0)
            events.append(ev2)
            components.append(ComponentOutput(
                component="retrieval",
                run_id=run_id,
                stage_index=2,
                scope_record_ids=scoped_df["record_id"].tolist() if "record_id" in scoped_df.columns else [],
                scope_record_count=len(scoped_df),
                scope_artifact_hash=_df_hash(scoped_df),
            ))

            # Stage 3: Fact extraction
            ev3, fact_df, t0 = self._stage_fact_extract(run_id, query, scoped_df, ev2.event_id, t0)
            events.append(ev3)
            components.append(ComponentOutput(
                component="extraction",
                run_id=run_id,
                stage_index=3,
                extraction_row_count=len(fact_df),
                extraction_artifact_hash=_df_hash(fact_df),
            ))

            # Stage 4: Aggregation
            ev4, agg_result, agg_plan, t0 = self._stage_aggregate(run_id, query, fact_df, ev3.event_id, t0)
            events.append(ev4)
            answer = agg_result
            components.append(ComponentOutput(
                component="aggregation",
                run_id=run_id,
                stage_index=4,
                aggregation_plan=agg_plan,
                aggregation_result=agg_result,
                aggregation_artifact_hash=_dict_hash({"result": agg_result, "plan": agg_plan}),
            ))

            # Stage 5: Validation
            ev5, is_correct, loss, t0 = self._stage_validate(
                run_id, query, agg_result, gold_answer, ev4.event_id, t0
            )
            events.append(ev5)
            components.append(ComponentOutput(
                component="validation",
                run_id=run_id,
                stage_index=5,
                validation_passed=is_correct,
                validation_message=ev5.message,
            ))

            # Stage 6: Persist
            ev6, artifact_refs, t0 = self._stage_persist(
                run_id, run_dir, query, events, components, scoped_df, fact_df, agg_result, gold_answer, t0
            )
            events.append(ev6)

            # Finalize run
            latency_ms = (time.monotonic() - (t0 - time.monotonic() + t0)) * 1000
            completed_at = datetime.now(timezone.utc)
            run = run.model_copy(update={
                "status": RunStatus.COMPLETED,
                "answer": agg_result,
                "gold_answer_value": gold_answer.answer_value if gold_answer else None,
                "is_correct": is_correct,
                "loss": loss,
                "latency_ms": float((completed_at - started_at).total_seconds() * 1000),
                "completed_at": completed_at,
                "artifact_references": artifact_refs,
            })

        except Exception as e:
            error = str(e)
            error_event = TraceEvent(
                run_id=run_id,
                stage="error",
                event_type=TraceEventType.ERROR,
                message=error,
                structured_payload={"error_type": type(e).__name__},
            )
            events.append(error_event)
            run = run.model_copy(update={
                "status": RunStatus.FAILED,
                "error_message": error,
                "completed_at": datetime.now(timezone.utc),
            })

        # Always save trace
        self._save_trace(run_dir, events)

        return run, events, components

    def _stage_query_load(
        self, run_id: str, query: QuerySpec, t0: float
    ) -> tuple[TraceEvent, float]:
        t1 = time.monotonic()
        event = TraceEvent(
            run_id=run_id,
            stage="query_load",
            event_type=TraceEventType.QUERY_LOAD,
            message=f"Loaded QuerySpec query_id={query.query_id} family={query.family.value}",
            structured_payload={
                "query_id": query.query_id,
                "family": query.family.value,
                "world_id": query.world_id,
                "template_id": query.template_id,
            },
            input_artifact_hash=_dict_hash({"query_id": query.query_id}),
            output_artifact_hash=_dict_hash(query.model_dump(mode="json")),
            duration_ms=(t1 - t0) * 1000,
        )
        return event, t1

    def _stage_scope_enumerate(
        self, run_id: str, query: QuerySpec, df: pd.DataFrame, parent_id: str, t0: float
    ) -> tuple[TraceEvent, pd.DataFrame, float]:
        t1 = time.monotonic()
        mask = compiler.to_pandas_mask(query.scope_predicate, df)
        scoped_df = df[mask].copy()
        t2 = time.monotonic()
        event = TraceEvent(
            run_id=run_id,
            parent_event_id=parent_id,
            stage="scope_enumerate",
            event_type=TraceEventType.SCOPE_ENUMERATE,
            record_count_in=len(df),
            record_count_out=len(scoped_df),
            message=f"Scope: {len(df)} -> {len(scoped_df)} records",
            structured_payload={
                "predicate_kind": query.scope_predicate.kind,
                "in_count": len(df),
                "out_count": len(scoped_df),
            },
            input_artifact_hash=_df_hash(df),
            output_artifact_hash=_df_hash(scoped_df),
            duration_ms=(t2 - t1) * 1000,
        )
        return event, scoped_df, t2

    def _stage_fact_extract(
        self, run_id: str, query: QuerySpec, scoped_df: pd.DataFrame, parent_id: str, t0: float
    ) -> tuple[TraceEvent, pd.DataFrame, float]:
        t1 = time.monotonic()
        # Select only specified fields
        available = set(scoped_df.columns)
        select_fields = [f for f in query.fact_spec.fields if f in available]
        if "record_id" not in select_fields and "record_id" in available:
            select_fields = ["record_id"] + select_fields
        fact_df = scoped_df[select_fields].copy()
        t2 = time.monotonic()
        event = TraceEvent(
            run_id=run_id,
            parent_event_id=parent_id,
            stage="fact_extract",
            event_type=TraceEventType.FACT_EXTRACT,
            record_count_in=len(scoped_df),
            record_count_out=len(fact_df),
            message=f"Extracted {len(select_fields)} fields from {len(fact_df)} records",
            structured_payload={
                "selected_fields": select_fields,
                "derived_fields": [d.name for d in query.fact_spec.derived_fields],
            },
            input_artifact_hash=_df_hash(scoped_df),
            output_artifact_hash=_df_hash(fact_df),
            duration_ms=(t2 - t1) * 1000,
        )
        return event, fact_df, t2

    def _stage_aggregate(
        self, run_id: str, query: QuerySpec, fact_df: pd.DataFrame, parent_id: str, t0: float
    ) -> tuple[TraceEvent, Any, dict, float]:
        t1 = time.monotonic()
        
        result, contributing_ids, metadata = pandas_eval._aggregate(query.aggregation_spec, fact_df, query)
        agg_result = result
        
        agg_plan = {
            "aggregation_kind": query.aggregation_spec.kind,
            "eligible_count": len(fact_df),
            "metadata": metadata,
        }
        t2 = time.monotonic()
        event = TraceEvent(
            run_id=run_id,
            parent_event_id=parent_id,
            stage="aggregate",
            event_type=TraceEventType.AGGREGATE,
            record_count_in=len(fact_df),
            message=f"Aggregation: {query.aggregation_spec.kind} -> {agg_result!r}",
            structured_payload=agg_plan,
            input_artifact_hash=_df_hash(fact_df),
            output_artifact_hash=_dict_hash({"result": str(agg_result)}),
            duration_ms=(t2 - t1) * 1000,
        )
        return event, agg_result, agg_plan, t2

    def _stage_validate(
        self,
        run_id: str,
        query: QuerySpec,
        answer: Any,
        gold: Optional[GoldAnswer],
        parent_id: str,
        t0: float,
    ) -> tuple[TraceEvent, Optional[bool], Optional[float], float]:
        t1 = time.monotonic()
        is_correct = None
        loss = None
        msg = "No gold answer available for comparison"
        payload: dict[str, Any] = {"pipeline_answer": str(answer)}

        if gold is not None:
            payload["gold_answer"] = str(gold.answer_value)
            # Numeric comparison
            try:
                a = float(answer) if answer is not None else None
                g = float(gold.answer_value) if gold.answer_value is not None else None
                if a is not None and g is not None:
                    diff = abs(a - g)
                    is_correct = diff <= gold.tolerance
                    loss = diff / (abs(g) + 1e-9)
                    msg = f"Answer={a}, Gold={g}, Diff={diff:.6f}, Correct={is_correct}"
                else:
                    is_correct = str(answer) == str(gold.answer_value)
                    loss = 0.0 if is_correct else 1.0
                    msg = f"Answer={answer!r}, Gold={gold.answer_value!r}, Correct={is_correct}"
            except (TypeError, ValueError):
                is_correct = str(answer) == str(gold.answer_value)
                loss = 0.0 if is_correct else 1.0
                msg = f"Non-numeric comparison: Correct={is_correct}"

        t2 = time.monotonic()
        payload.update({"is_correct": is_correct, "loss": loss})
        event = TraceEvent(
            run_id=run_id,
            parent_event_id=parent_id,
            stage="validate",
            event_type=TraceEventType.VALIDATE,
            message=msg,
            structured_payload=payload,
            duration_ms=(t2 - t1) * 1000,
        )
        return event, is_correct, loss, t2

    def _stage_persist(
        self,
        run_id: str,
        run_dir: Path,
        query: QuerySpec,
        events: list[TraceEvent],
        components: list[ComponentOutput],
        scoped_df: pd.DataFrame,
        fact_df: pd.DataFrame,
        agg_result: Any,
        gold: Optional[GoldAnswer],
        t0: float,
    ) -> tuple[TraceEvent, dict[str, str], float]:
        t1 = time.monotonic()
        refs: dict[str, str] = {}

        # Save scope output
        scope_path = run_dir / "scope_output.parquet"
        scoped_df.to_parquet(scope_path, index=False)
        refs["scope_output"] = str(scope_path)

        # Save extraction
        extract_path = run_dir / "extraction.parquet"
        fact_df.to_parquet(extract_path, index=False)
        refs["extraction"] = str(extract_path)

        # Save aggregation result
        agg_path = run_dir / "aggregation_result.json"
        agg_path.write_text(json.dumps({"result": str(agg_result)}, default=str), encoding="utf-8")
        refs["aggregation_result"] = str(agg_path)

        # Save gold comparison
        if gold:
            gold_path = run_dir / "gold_answer.json"
            gold_path.write_text(json.dumps(gold.model_dump(mode="json"), default=str), encoding="utf-8")
            refs["gold_answer"] = str(gold_path)

        # Save query spec
        query_path = run_dir / "query_spec.json"
        query_path.write_text(json.dumps(query.model_dump(mode="json"), default=str), encoding="utf-8")
        refs["query_spec"] = str(query_path)

        t2 = time.monotonic()
        event = TraceEvent(
            run_id=run_id,
            stage="persist",
            event_type=TraceEventType.PERSIST,
            message=f"Saved {len(refs)} artifacts to {run_dir}",
            structured_payload={"artifact_refs": refs},
            duration_ms=(t2 - t1) * 1000,
        )
        return event, refs, t2

    def _save_trace(self, run_dir: Path, events: list[TraceEvent]) -> None:
        trace_path = run_dir / "trace.jsonl"
        with open(trace_path, "wb") as f:
            for ev in events:
                f.write(orjson.dumps(ev.model_dump(mode="json")))
                f.write(b"\n")


def _df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame's content."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    if len(df) > 0:
        h.update(str(df.values.tolist()[:100]).encode())
    return h.hexdigest()[:32]


def _dict_hash(d: dict) -> str:
    """Stable hash of a dict."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:32]
