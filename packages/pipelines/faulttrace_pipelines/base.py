"""
Abstract base class for all FaultTrace-RAG pipelines (P0–P5).

Each pipeline must implement:
  - pipeline_id: str
  - provider_id: str
  - run(query, df, gold_answer, parquet_path) -> (PipelineRun, list[TraceEvent], list[ComponentOutput])

Design decisions:
- The base class owns timing, error wrapping, and artifact persistence boilerplate.
- Subclasses only implement _execute() which must return the raw answer value.
- Token estimates are best-effort: LLM pipelines set them, deterministic pipelines leave them at 0.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from faulttrace_core.models import (
    ComponentOutput,
    GoldAnswer,
    PipelineRun,
    QuerySpec,
    RunStatus,
    TraceEvent,
    TraceEventType,
)


class AbstractPipeline(ABC):
    """Base class for all FaultTrace-RAG pipelines."""

    pipeline_id: str
    provider_id: str

    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        self.artifacts_dir = Path(artifacts_dir)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        query: QuerySpec,
        df: pd.DataFrame,
        gold_answer: Optional[GoldAnswer] = None,
        parquet_path: Optional[Path] = None,
    ) -> tuple[PipelineRun, list[TraceEvent], list[ComponentOutput]]:
        """
        Execute the pipeline and return (run, trace_events, component_outputs).
        This method handles timing, error catching, and artifact persistence.
        """
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        trace_events: list[TraceEvent] = []
        component_outputs: list[ComponentOutput] = []
        answer_value: Any = None
        token_in = 0
        token_out = 0
        error_message: Optional[str] = None
        error_stage: Optional[str] = None

        t0 = time.perf_counter()
        try:
            answer_value, trace_events, component_outputs, token_in, token_out = (
                self._execute(run_id, query, df, parquet_path, gold_answer)
            )
            status = RunStatus.COMPLETED
        except Exception as exc:
            status = RunStatus.FAILED
            error_message = str(exc)
            error_stage = getattr(exc, "stage", "unknown")

        latency_ms = (time.perf_counter() - t0) * 1000
        completed_at = datetime.now(timezone.utc)

        # Compare to gold
        is_correct: Optional[bool] = None
        is_within_tolerance: Optional[bool] = None
        loss: Optional[float] = None
        gold_answer_value = None

        if gold_answer is not None and answer_value is not None:
            gold_answer_value = gold_answer.answer_value
            try:
                from faulttrace_gold.validator import _results_agree
                is_correct = _results_agree(
                    answer_value,
                    gold_answer.answer_value,
                    gold_answer.tolerance,
                )
                is_within_tolerance = is_correct
                if isinstance(answer_value, (int, float)) and isinstance(
                    gold_answer.answer_value, (int, float)
                ):
                    loss = abs(float(answer_value) - float(gold_answer.answer_value))
                else:
                    loss = 0.0 if is_correct else 1.0
            except Exception:
                pass

        # Persist artifacts
        artifact_references = self._persist_artifacts(
            run_id=run_id,
            query=query,
            answer_value=answer_value,
            gold_answer=gold_answer,
            trace_events=trace_events,
            component_outputs=component_outputs,
        )

        pipeline_run = PipelineRun(
            run_id=run_id,
            query_id=str(query.query_id),
            pipeline_id=self.pipeline_id,
            provider_id=self.provider_id,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            answer=answer_value,
            raw_answer=answer_value,
            gold_answer_value=gold_answer_value,
            is_correct=is_correct,
            is_within_tolerance=is_within_tolerance,
            loss=loss,
            latency_ms=latency_ms,
            token_estimate_input=token_in,
            token_estimate_output=token_out,
            error_message=error_message,
            error_stage=error_stage,
            artifact_references=artifact_references,
        )
        
        pipeline_run.config_hash = pipeline_run.compute_config_hash(query)
        
        # Certification
        from faulttrace_core.models import AnswerPolicyConfig, CoverageDecision
        from faulttrace_pipelines.coverage_adapters import extract_coverage_observations
        from faulttrace_pipelines.certification import CertificationEngine
        
        obs = extract_coverage_observations(pipeline_run, trace_events, df)
        policy_config = AnswerPolicyConfig(policy_id="strict_exact_v1")
        cert_engine = CertificationEngine(policy=policy_config)
        cert = cert_engine.certify(pipeline_run, query, obs)
        
        pipeline_run.certificate_id = cert.certificate_id
        pipeline_run.certificate_hash = cert.certificate_hash
        pipeline_run.policy_decision = cert.decision.value
        
        if cert.decision == CoverageDecision.CERTIFIED:
            pipeline_run.final_presented_answer = answer_value
        else:
            pipeline_run.final_presented_answer = None
            pipeline_run.abstention_reason = " | ".join(c.value for c in cert.reason_codes)

        return pipeline_run, trace_events, component_outputs


    # ── Subclass interface ──────────────────────────────────────────────────

    @abstractmethod
    def _execute(
        self,
        run_id: str,
        query: QuerySpec,
        df: pd.DataFrame,
        parquet_path: Optional[Path],
        gold_answer: Optional[GoldAnswer],
    ) -> tuple[Any, list[TraceEvent], list[ComponentOutput], int, int]:
        """
        Run the actual pipeline logic.

        Returns:
            (answer_value, trace_events, component_outputs, token_in, token_out)
        """
        ...

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _make_event(
        self,
        run_id: str,
        stage: str,
        event_type: TraceEventType,
        message: str,
        record_count_in: Optional[int] = None,
        record_count_out: Optional[int] = None,
        duration_ms: Optional[float] = None,
        payload: Optional[dict] = None,
        parent_event_id: Optional[str] = None,
    ) -> TraceEvent:
        return TraceEvent(
            event_id=str(uuid4()),
            run_id=run_id,
            parent_event_id=parent_event_id,
            stage=stage,
            event_type=event_type,
            message=message,
            record_count_in=record_count_in,
            record_count_out=record_count_out,
            duration_ms=duration_ms,
            structured_payload=payload or {},
        )

    def _persist_artifacts(
        self,
        run_id: str,
        query: QuerySpec,
        answer_value: Any,
        gold_answer: Optional[GoldAnswer],
        trace_events: list[TraceEvent],
        component_outputs: list[ComponentOutput],
    ) -> dict[str, str]:
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        refs: dict[str, str] = {}

        # Query spec
        qspec_path = run_dir / "query_spec.json"
        qspec_path.write_text(query.model_dump_json(indent=2), encoding="utf-8")
        refs["query_spec"] = str(qspec_path)

        # Answer
        ans_path = run_dir / "aggregation_result.json"
        ans_path.write_text(
            json.dumps({"answer": answer_value, "pipeline_id": self.pipeline_id}, default=str),
            encoding="utf-8"
        )
        refs["aggregation_result"] = str(ans_path)

        # Gold answer
        if gold_answer is not None:
            gold_path = run_dir / "gold_answer.json"
            gold_path.write_text(gold_answer.model_dump_json(indent=2), encoding="utf-8")
            refs["gold_answer"] = str(gold_path)

        # Trace JSONL
        trace_path = run_dir / "trace.jsonl"
        lines = [ev.model_dump_json() for ev in trace_events]
        trace_path.write_text("\n".join(lines), encoding="utf-8")
        refs["trace"] = str(trace_path)

        # Auto-detect standard pipeline artifacts
        scope_path = run_dir / "scope_output.parquet"
        if scope_path.exists():
            refs["scope_enumerate"] = str(scope_path)
            
        extract_path = run_dir / "extraction.parquet"
        if extract_path.exists():
            refs["fact_extract"] = str(extract_path)

        return refs
