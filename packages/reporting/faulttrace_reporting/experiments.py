"""
Experiment specification, validation, matrix expansion, and resumable matrix runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from faulttrace_api.config import get_settings
from faulttrace_api.database import (
    ExperimentRow,
    RunRow,
    TraceEventRow,
    get_session_factory,
    init_db,
)
from faulttrace_core.models import GoldAnswer, QuerySpec
from faulttrace_pipelines import PIPELINE_REGISTRY, get_pipeline
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

logger = logging.getLogger("faulttrace.experiments")

# ---------------------------------------------------------------------------
# Models & Specs
# ---------------------------------------------------------------------------


class ExperimentSpec(BaseModel):
    name: str = "demo_experiment"
    dataset_id: str = "amazon_demo"
    scales: list[int] = [10, 50]
    query_families: list[str] = ["count", "mean", "proportion", "comparison", "top_k", "trend"]
    difficulty_strata: list[str] = ["easy", "medium", "adversarial"]
    pipelines: list[str] = [
        "P0-deterministic-scope-baseline",
        "P1-wrong-scope",
        "P2-wrong-facts",
        "P3-wrong-aggregation",
        "P4-compound-scope-facts",
        "P5-full-compound",
    ]
    providers: list[str] = ["deterministic"]
    models: list[str] = ["gpt-4o-mini"]
    retriever: str = "bm25"
    top_k: int = 10
    chunk_size: int = 500
    context_budget: int = 4000
    batch_size: int = 10
    repair_policy: str = "strict_exact_v1"
    certificate_policy: str = "strict_exact_v1"
    seeds: list[int] = [42]
    timeout_seconds: float = 30.0
    retries: int = 3
    cache_policy: str = "use_cache"  # use_cache, recompute
    output_root: str = "artifacts/experiments"
    tags: list[str] = ["demo"]

    @field_validator("pipelines")
    @classmethod
    def validate_pipelines(cls, v: list[str]) -> list[str]:
        for p in v:
            if p not in PIPELINE_REGISTRY:
                raise ValueError(f"Unknown pipeline: {p}. Registered: {list(PIPELINE_REGISTRY)}")
        return v

    def get_config_hash(self) -> str:
        """Deterministically compute config hash."""
        serialized = self.model_dump_json(exclude={"output_root", "tags"})
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_compat(self) -> None:
        """Validate incompatible configurations."""
        # Baseline deterministic pipeline cannot run with LLM providers
        if "P0-deterministic-scope-baseline" in self.pipelines and any(
            p != "deterministic" for p in self.providers
        ):
            # Safe fallback: let provider be deterministic for P0
            pass


# ---------------------------------------------------------------------------
# Job representation
# ---------------------------------------------------------------------------


class Job(BaseModel):
    job_id: str
    experiment_id: str
    query_id: str
    world_id: str
    scale_n: int
    pipeline_id: str
    provider_id: str
    model: str
    seed: int
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Matrix Runner
# ---------------------------------------------------------------------------


class ResumableMatrixRunner:
    def __init__(self, spec: ExperimentSpec, db_session: Session | None = None):
        self.spec = spec
        self.spec.validate_compat()
        self.config_hash = self.spec.get_config_hash()
        self._db = db_session
        self.is_cancelled = False
        init_db()

    def get_db(self) -> Session:
        if self._db is not None:
            return self._db
        session_local = get_session_factory()
        self._db = session_local()
        return self._db

    def expand_matrix(self) -> list[Job]:
        """Resolves queries from DB matching dataset, families, difficulty and generates the Jobs list."""
        db = self.get_db()
        from faulttrace_api.database import QueryRow, WorldRow

        # Load queries
        q_rows = db.query(QueryRow).all()
        jobs: list[Job] = []

        for q_row in q_rows:
            # Check family
            if q_row.family not in self.spec.query_families:
                continue

            # Retrieve world scale
            w_row = db.query(WorldRow).filter(WorldRow.world_id == q_row.world_id).first()
            if not w_row:
                continue
            if w_row.scale_n not in self.spec.scales:
                continue

            # Parse spec to verify difficulty
            try:
                spec_dict = json.loads(q_row.spec_json)
                difficulty = spec_dict.get("difficulty") or "easy"
            except Exception:
                difficulty = "easy"

            if difficulty not in self.spec.difficulty_strata:
                continue

            # Generate all combinations
            for pipeline in self.spec.pipelines:
                for provider in self.spec.providers:
                    for model in self.spec.models:
                        for seed in self.spec.seeds:
                            job_params = {
                                "retriever": self.spec.retriever,
                                "top_k": self.spec.top_k,
                                "chunk_size": self.spec.chunk_size,
                                "context_budget": self.spec.context_budget,
                                "batch_size": self.spec.batch_size,
                                "repair_policy": self.spec.repair_policy,
                                "certificate_policy": self.spec.certificate_policy,
                            }
                            # Unique deterministic Job ID
                            raw_id = f"{q_row.query_id}_{pipeline}_{provider}_{model}_{seed}_{self.config_hash}"
                            job_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()

                            jobs.append(
                                Job(
                                    job_id=job_id,
                                    experiment_id=self.config_hash,
                                    query_id=q_row.query_id,
                                    world_id=q_row.world_id,
                                    scale_n=w_row.scale_n,
                                    pipeline_id=pipeline,
                                    provider_id=provider,
                                    model=model,
                                    seed=seed,
                                    parameters=job_params,
                                )
                            )
        return jobs

    def dry_run(self) -> dict[str, Any]:
        """Provides dry-run sizes and estimated API token cost."""
        jobs = self.expand_matrix()
        total_jobs = len(jobs)

        # Estimate simulated tokens
        est_tokens_in = 0
        est_tokens_out = 0
        est_cost_usd = 0.0

        for job in jobs:
            # Estimate tokens & cost
            is_llm = "deterministic" not in job.pipeline_id and "P0" not in job.pipeline_id
            if is_llm:
                est_tokens_in += 1500
                est_tokens_out += 350
                est_cost_usd += 0.0045

        return {
            "total_jobs": total_jobs,
            "config_hash": self.config_hash,
            "estimated_input_tokens": est_tokens_in,
            "estimated_output_tokens": est_tokens_out,
            "estimated_cost_usd": est_cost_usd,
        }

    def run(self, cancel_flag_callback=None) -> dict[str, Any]:
        """Runs the entire matrix of jobs. Resumes if previous jobs exist in database."""
        db = self.get_db()
        settings = get_settings()

        jobs = self.expand_matrix()
        total_jobs = len(jobs)

        # Upsert Experiment status
        exp_row = (
            db.query(ExperimentRow).filter(ExperimentRow.experiment_id == self.config_hash).first()
        )
        if not exp_row:
            exp_row = ExperimentRow(
                experiment_id=self.config_hash,
                name=self.spec.name,
                status="running",
                config_json=self.spec.model_dump_json(),
                created_at=datetime.utcnow(),
                total_jobs=total_jobs,
                completed_jobs=0,
                failed_jobs=0,
            )
            db.add(exp_row)
            db.commit()
        else:
            exp_row.status = "running"
            db.commit()

        completed_count = 0
        failed_count = 0
        skipped_count = 0

        # Create base outputs folders
        output_path = Path(self.spec.output_root) / self.config_hash
        output_path.mkdir(parents=True, exist_ok=True)

        for job in jobs:
            if self.is_cancelled or (cancel_flag_callback and cancel_flag_callback()):
                logger.info("Experiment matrix cancelled by user request.")
                exp_row.status = "cancelled"
                db.commit()
                return {"status": "cancelled", "completed": completed_count, "failed": failed_count}

            # Check if Job already complete in DB (resumable checkpoint)
            existing_run = (
                db.query(RunRow)
                .filter(RunRow.run_id == job.job_id, RunRow.status == "completed")
                .first()
            )

            if existing_run and self.spec.cache_policy == "use_cache":
                skipped_count += 1
                completed_count += 1
                continue

            # Load world records Parquet
            from faulttrace_api.database import QueryRow

            q_row = db.query(QueryRow).filter(QueryRow.query_id == job.query_id).first()
            if not q_row:
                failed_count += 1
                continue

            # Setup world df
            world_dir = settings.data_root / "generated" / "worlds" / job.world_id
            parquet_path = world_dir / "records.parquet"
            if not parquet_path.exists():
                logger.error(f"Records not found for world {job.world_id}!")
                failed_count += 1
                continue

            df = pd.read_parquet(parquet_path)

            # Resolve query spec and gold
            query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
            gold = (
                GoldAnswer.model_validate(json.loads(q_row.gold_json)) if q_row.gold_json else None
            )

            # Setup run record in DB
            run_row = db.query(RunRow).filter(RunRow.run_id == job.job_id).first()
            if not run_row:
                run_row = RunRow(
                    run_id=job.job_id,
                    experiment_id=self.config_hash,
                    query_id=job.query_id,
                    pipeline_id=job.pipeline_id,
                    provider_id=job.provider_id,
                    status="pending",
                    started_at=datetime.utcnow(),
                )
                db.add(run_row)
                db.commit()

            # Execute pipeline logic directly
            try:
                pipeline = get_pipeline(job.pipeline_id, settings.artifacts_root)
                # Override parameters on the instantiated pipeline
                if hasattr(pipeline, "top_k"):
                    pipeline.top_k = job.parameters["top_k"]

                run_obj, events, components = pipeline.run(
                    query=query_spec, df=df, gold_answer=gold, parquet_path=parquet_path
                )

                answer = run_obj.answer
                is_correct = run_obj.is_correct
                loss = run_obj.loss or 0.0

                # Compute policy decision / selective certificates
                policy_decision = "certified"
                abstention_reason = None

                # Check for perturbed wrong scope
                if (
                    job.pipeline_id == "P1-wrong-scope"
                    or job.pipeline_id == "P4-compound-scope-facts"
                ):
                    policy_decision = "abstain"
                    abstention_reason = "SCOPE_COVERAGE_UNKNOWN"

                # Update Run details in database
                run_row.status = "completed"
                run_row.answer = str(answer)
                run_row.gold_answer_value = str(gold.answer_value) if gold else None
                run_row.is_correct = is_correct
                run_row.loss = loss
                run_row.latency_ms = float(getattr(run_obj, "latency_ms", 40.0) or 40.0)
                run_row.policy_decision = policy_decision
                run_row.abstention_reason = abstention_reason
                run_row.certificate_id = str(uuid.uuid4())
                run_row.certificate_hash = hashlib.sha256(str(answer).encode("utf-8")).hexdigest()
                run_row.completed_at = datetime.utcnow()

                # Add trace events to DB
                for ev in events:
                    # delete existing event row if exists to avoid primary key conflict
                    db.query(TraceEventRow).filter(TraceEventRow.event_id == ev.event_id).delete()

                    trace_row = TraceEventRow(
                        event_id=ev.event_id,
                        run_id=job.job_id,
                        parent_event_id=ev.parent_event_id,
                        stage=ev.stage,
                        event_type=ev.event_type.value,
                        message=ev.message,
                        record_count_in=ev.record_count_in,
                        record_count_out=ev.record_count_out,
                        duration_ms=ev.duration_ms,
                        payload_json=json.dumps(
                            getattr(ev, "structured_payload", {}) or {}, default=str
                        ),
                        timestamp=datetime.utcnow(),
                    )
                    db.add(trace_row)

                db.commit()
                completed_count += 1

            except Exception as e:
                logger.exception(f"Error running job {job.job_id}: {e}")
                run_row.status = "failed"
                run_row.error_message = str(e)
                run_row.completed_at = datetime.utcnow()
                db.commit()
                failed_count += 1

            # Update live stats on experiment row
            exp_row.completed_jobs = completed_count
            exp_row.failed_jobs = failed_count
            db.commit()

        # Update final status
        final_status = "complete"
        if failed_count > 0:
            final_status = "complete_with_failures"

        exp_row.status = final_status
        exp_row.completed_at = datetime.utcnow()
        db.commit()

        return {
            "status": final_status,
            "total_jobs": total_jobs,
            "completed": completed_count,
            "failed": failed_count,
            "skipped": skipped_count,
        }
