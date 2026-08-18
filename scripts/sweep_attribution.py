import json
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import ast

import faulttrace_api.database as db_models
import pandas as pd
from faulttrace_api.config import get_settings
from faulttrace_api.database import RunRow, get_session_factory, init_db
from faulttrace_core.models import GoldAnswer, PipelineRun, QuerySpec
from faulttrace_pipelines.batch_attribution import BatchAttributionRunner
from faulttrace_reporting import ExperimentSpec, ResumableMatrixRunner


def ensure_datasets():
    print("Seeding missing world datasets...")
    settings = get_settings()
    str(settings.database_url)

    cmd = [
        sys.executable,
        "-m",
        "faulttrace_data.cli",
        "data",
        "seed",
        "--seed",
        "42",
        "--scales",
        "10,50,200,1000,2000,5000",
    ]
    subprocess.run(cmd, check=True)

    print("Generating queries for all seeded worlds...")
    for n in [10, 50, 200, 1000, 2000, 5000]:
        world_id = f"world_s42_n{n}"
        qcmd = [
            sys.executable,
            "-m",
            "faulttrace_data.cli",
            "query",
            "generate",
            "--world-id",
            world_id,
        ]
        subprocess.run(qcmd, check=True)

    print("Inserting data into database manually (bypassing missing repositories)...")
    session_local = get_session_factory()
    db = session_local()
    import json
    from datetime import datetime

    from faulttrace_api.database import QueryRow, WorldRow
    from faulttrace_gold.duckdb_engine import DuckDBEvaluator

    for n in [10, 50, 200, 1000, 2000, 5000]:
        world_id = f"world_s42_n{n}"
        world_dir = settings.data_root / "generated" / "worlds" / world_id
        parquet_path = world_dir / "records.parquet"
        if not parquet_path.exists():
            continue

        manifest_path = world_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Insert World
        if not db.query(WorldRow).filter(WorldRow.world_id == world_id).first():
            w = WorldRow(
                world_id=world_id,
                dataset_id="amazon_demo",
                seed=42,
                scale_n=n,
                record_ids_hash=manifest["parquet_hash"],
                manifest_path=str(manifest_path.relative_to(settings.data_root)),
                created_at=datetime.utcnow(),
            )
            db.add(w)

        # Insert Queries
        q_path = settings.artifacts_root / "queries" / f"queries_{world_id}.jsonl"
        if q_path.exists():
            pd.read_parquet(parquet_path)
            evaluator = DuckDBEvaluator()

            with open(q_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    q_dict = json.loads(line)
                    q_id = q_dict["query_id"]
                    if not db.query(QueryRow).filter(QueryRow.query_id == q_id).first():
                        spec = QuerySpec.model_validate(q_dict)
                        # Compute Gold Answer!
                        try:
                            t0 = time.time()
                            eval_res = evaluator.evaluate(spec, parquet_path)
                            (time.time() - t0) * 1000
                            gold = GoldAnswer(
                                query_id=q_id,
                                world_id=world_id,
                                answer_value=eval_res["result"],
                                eligible_record_count=eval_res["eligible_count"],
                                contributing_record_ids=eval_res["contributing_ids"],
                                evidence_hash=manifest["parquet_hash"],
                                duckdb_result=eval_res["result"],
                            )
                            gold_json = gold.model_dump_json()
                        except Exception as e:
                            print(f"Error computing gold answer for {q_id}: {e}")
                            gold_json = None

                        q = QueryRow(
                            query_id=q_id,
                            world_id=world_id,
                            family=spec.family.value,
                            natural_language_question=spec.natural_language_question,
                            template_id=spec.template_id,
                            version="1.0",
                            spec_json=spec.model_dump_json(),
                            gold_json=gold_json,
                            created_at=datetime.utcnow(),
                        )
                        db.add(q)

    db.commit()
    db.close()
    print("Datasets, queries, and gold answers successfully seeded and registered.")


def run_experiment(retriever: str):
    print("\n======================================")
    print(f"Running sweep for retriever: {retriever}")
    print("======================================")

    spec = ExperimentSpec(
        name=f"sweep_attribution_{retriever}",
        dataset_id="amazon_demo",
        scales=[10, 50, 200, 1000, 2000, 5000],
        query_families=["count", "mean", "proportion", "comparison", "top_k", "trend"],
        difficulty_strata=["easy", "medium", "adversarial"],
        pipelines=[
            "P1-direct-bm25",
            "P2-direct-dense",
            "P4-full-scope-mer",
            "P5-certified-mer-repair",
        ],
        providers=["deterministic"],
        models=["default"],
        retriever=retriever,
        top_k=10,
        chunk_size=500,
        context_budget=4000,
        batch_size=10,
        seeds=[42],
    )

    # Initialize matrix runner
    runner = ResumableMatrixRunner(spec)

    # Dry run stats
    plan = runner.dry_run()
    print(f"Dry run total jobs for {retriever}: {plan['total_jobs']}")

    # Execute Matrix
    result = runner.run()
    print(f"Experiment {retriever} execution result: {result}")

    return spec.get_config_hash()


def generate_attribution_parquets(exp_hash: str, batch_id: str):
    print(f"Gathering runs for attribution from experiment {exp_hash}...")
    session_local = get_session_factory()
    db = session_local()
    settings = get_settings()

    run_rows = (
        db.query(RunRow)
        .filter(RunRow.experiment_id == exp_hash, RunRow.status == "completed")
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

    print(f"Processing {len(runs_to_process)} runs through BatchAttributionRunner...")
    batch_runner = BatchAttributionRunner(
        artifacts_dir=settings.artifacts_root / "batch_diagnostics"
    )
    summary = batch_runner.run_batch(batch_id, runs_to_process)

    print(f"Batch Attribution {batch_id} summary: {json.dumps(summary, indent=2)}")


def main():
    # 1. Init DB & Ensure data
    init_db()
    ensure_datasets()

    # 2. Sweep over retrievers
    for retriever in ["bm25", "dense"]:
        exp_hash = run_experiment(retriever)

        # 3. Generate counterfactual lattice values as Parquet
        batch_id = f"attribution_{retriever}_{exp_hash}"
        generate_attribution_parquets(exp_hash, batch_id)


if __name__ == "__main__":
    main()
