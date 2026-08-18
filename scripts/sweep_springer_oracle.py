import ast
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import faulttrace_api.database as db_models
import pandas as pd
from faulttrace_api.config import get_settings
from faulttrace_api.database import RunRow, get_session_factory
from faulttrace_core.models import GoldAnswer, PipelineRun, QuerySpec
from faulttrace_pipelines.batch_attribution import BatchAttributionRunner
from faulttrace_reporting import ExperimentSpec, ResumableMatrixRunner


def run_springer_sweep():
    print("======================================")
    print("Running Springer Oracle Replacement Sweep")
    print("======================================")

    pipelines = [
        "P0-deterministic-scope-baseline",
        "P1-wrong-scope",
        "P2-wrong-facts",
        "P3-wrong-aggregation",
        "P4-compound-scope-facts",
        "P5-full-compound",
        "gnn-extractor",
    ]

    spec = ExperimentSpec(
        name="sweep_springer_oracle",
        dataset_id="springer_toc",
        scales=[10, 50, 200, 1000, 2000, 5000],
        query_families=["count", "mean", "proportion", "comparison", "top_k", "trend"],
        difficulty_strata=["easy", "medium", "adversarial"],
        pipelines=pipelines,
        providers=["deterministic"],
        models=["default"],
        retriever="bm25",
        top_k=10,
        chunk_size=500,
        context_budget=4000,
        batch_size=10,
        seeds=[42],
    )

    runner = ResumableMatrixRunner(spec)

    jobs = runner.expand_matrix()
    print(f"Total jobs for Springer Sweep: {len(jobs)}")

    result = runner.run()
    print(f"Experiment execution result: {result}")

    print(f"Gathering runs for attribution from experiment {runner.config_hash}...")

    session_local = get_session_factory()
    db = session_local()
    settings = get_settings()

    run_rows = (
        db.query(RunRow)
        .filter(RunRow.experiment_id == runner.config_hash, RunRow.status == "completed")
        .all()
    )

    runs_to_process = []

    for run_row in run_rows:
        if not run_row.gold_answer_value:
            continue

        q_row = (
            db.query(db_models.QueryRow)
            .filter(db_models.QueryRow.query_id == run_row.query_id)
            .first()
        )
        if not q_row:
            continue

        query_spec = QuerySpec.model_validate(json.loads(q_row.spec_json))
        gold = GoldAnswer.model_validate(json.loads(q_row.gold_json))

        world_dir = settings.data_root / "generated" / "worlds" / query_spec.world_id
        parquet_path = world_dir / "records.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)

        raw = run_row.answer
        try:
            pipeline_answer = ast.literal_eval(raw)
        except Exception:
            pipeline_answer = raw

        parent_run = PipelineRun(
            run_id=run_row.run_id,
            query_id=run_row.query_id,
            pipeline_id=run_row.pipeline_id,
            provider_id=run_row.provider_id,
            started_at=run_row.started_at,
            completed_at=run_row.completed_at,
            status=run_row.status,
            answer=pipeline_answer,
            gold_answer_value=run_row.gold_answer_value,
            is_correct=run_row.is_correct,
            is_within_tolerance=run_row.is_correct,
            loss=run_row.loss,
            latency_ms=run_row.latency_ms,
            token_estimate_input=0,
            token_estimate_output=0,
            artifact_references=json.loads(run_row.artifact_refs_json)
            if run_row.artifact_refs_json
            else {},
        )

        runs_to_process.append((parent_run, query_spec, gold, df))

    db.close()

    print(f"Processing {len(runs_to_process)} runs through BatchAttributionRunner...")

    out_dir = Path("outputs/parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_runner = BatchAttributionRunner(artifacts_dir=out_dir)
    summary = batch_runner.run_batch(f"springer_{runner.config_hash}", runs_to_process)

    print(f"Batch Attribution summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    run_springer_sweep()
