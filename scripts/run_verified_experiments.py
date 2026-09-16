"""Run every checked-in experiment config against a clean immutable Track-M seed.

The output is a self-contained, ignored reproducibility directory. Source
documentation may quote results only from ``verified_results.json`` produced by
this script.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [
    REPOSITORY_ROOT / "apps" / "api",
    REPOSITORY_ROOT / "packages" / "core",
    REPOSITORY_ROOT / "packages" / "data",
    REPOSITORY_ROOT / "packages" / "gold",
    REPOSITORY_ROOT / "packages" / "pipelines",
    REPOSITORY_ROOT / "packages" / "reporting",
]
for source_root in reversed(SOURCE_ROOTS):
    sys.path.insert(0, str(source_root))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*arguments: str) -> str | None:
    process = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def source_state() -> dict[str, Any]:
    executable_paths = [
        "apps",
        "packages",
        "scripts",
        "configs",
        "docker",
        ".dockerignore",
        ".env.example",
        "docker-compose.yml",
        "Makefile",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements.lock.txt",
    ]
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *executable_paths],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    ).stdout
    untracked = git_value("ls-files", "--others", "--exclude-standard") or ""
    source_prefixes = ("apps/", "packages/", "scripts/", "configs/")
    root_source_files = {
        ".dockerignore",
        ".env.example",
        "Dockerfile",
        "docker-compose.yml",
        "Makefile",
        "pyproject.toml",
        "requirements.lock.txt",
    }
    untracked_source_paths = sorted(
        path
        for path in untracked.splitlines()
        if path in root_source_files or path.startswith(source_prefixes)
    )
    return {
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_scope": executable_paths,
        "untracked_source_paths": untracked_source_paths,
    }


def configure_isolated_environment(output_dir: Path) -> None:
    os.environ["FAULTTRACE_DATA_ROOT"] = str((output_dir / "data").resolve())
    os.environ["FAULTTRACE_ARTIFACTS_ROOT"] = str((output_dir / "runs").resolve())
    os.environ["FAULTTRACE_DB_PATH"] = str((output_dir / "registry.sqlite3").resolve())
    os.environ["FAULTTRACE_DATABASE_URL"] = ""
    os.environ["MPLBACKEND"] = "Agg"


def checked_in_configs() -> list[Path]:
    paths = [REPOSITORY_ROOT / "configs" / "experiments" / "demo.json"]
    paths.extend(sorted((REPOSITORY_ROOT / "packages" / "data" / "configs" / "experiments").glob("*.json")))
    return paths


def dataset_inventory(data_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def make_dataset_read_only(data_root: Path) -> None:
    for path in data_root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)


def collect_runs(db: Any, experiment_id: str, jobs: dict[str, Any]) -> list[dict[str, Any]]:
    from faulttrace_api.database import QueryRow, RunRow, WorldRow

    records: list[dict[str, Any]] = []
    for run in db.query(RunRow).filter(RunRow.experiment_id == experiment_id).all():
        query_row = db.query(QueryRow).filter(QueryRow.query_id == run.query_id).one()
        world_row = db.query(WorldRow).filter(WorldRow.world_id == query_row.world_id).one()
        query_payload = json.loads(query_row.spec_json)
        job = jobs[run.run_id]
        artifact_refs = json.loads(run.artifact_refs_json or "{}")
        provenance_value = artifact_refs.get("provenance")
        provenance_path = Path(provenance_value) if provenance_value else None
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path is not None and provenance_path.exists()
            else {}
        )
        certificate_value = artifact_refs.get("certificate")
        certificate_path = Path(certificate_value) if certificate_value else None
        certificate = (
            json.loads(certificate_path.read_text(encoding="utf-8"))
            if certificate_path is not None and certificate_path.exists()
            else {}
        )
        semantic_reason_codes = {
            "PROVENANCE_UNVERIFIABLE",
            "PROVENANCE_MISMATCH",
            "FACT_FIDELITY_UNKNOWN",
            "FACT_FIDELITY_BELOW_REQUIRED",
            "NUMERIC_FIDELITY_UNKNOWN",
            "NUMERIC_FIDELITY_BELOW_REQUIRED",
            "AGGREGATION_REPLAY_UNKNOWN",
            "AGGREGATION_REPLAY_MISMATCH",
        }
        reason_codes = [str(value) for value in certificate.get("reason_codes", [])]
        structural_blockers = [
            value
            for value in reason_codes
            if value not in semantic_reason_codes and value != "CERTIFIED"
        ]
        records.append(
            {
                "run_id": run.run_id,
                "experiment_id": experiment_id,
                "status": run.status,
                "dataset_id": job.dataset_id,
                "world_id": world_row.world_id,
                "world_record_ids_hash": world_row.record_ids_hash,
                "scale_n": world_row.scale_n,
                "query_id": run.query_id,
                "query_family": query_row.family,
                "difficulty": query_payload.get("difficulty"),
                "split": query_payload.get("split"),
                "query_spec_hash": job.query_spec_hash,
                "gold_answer_hash": job.gold_answer_hash,
                "pipeline_id": run.pipeline_id,
                "requested_provider_id": job.provider_id,
                "actual_provider_id": run.provider_id,
                "model_id": job.model,
                "seed": job.seed,
                "top_k": job.parameters["top_k"],
                "answer": run.answer,
                "gold_answer_value": run.gold_answer_value,
                "is_correct": run.is_correct,
                "loss": run.loss,
                "latency_ms": run.latency_ms,
                "policy_decision": run.policy_decision,
                "certificate_hash": run.certificate_hash,
                "certificate_policy_id": provenance.get("certificate_policy_id"),
                "certificate_assurance_scope": provenance.get(
                    "certificate_assurance_scope"
                ),
                "certificate_ratios": certificate.get("coverage_ratios", {}),
                "certificate_reason_codes": reason_codes,
                "structural_policy_counterfactual_decision": (
                    "certified" if not structural_blockers else "abstain"
                ),
                "config_hash": run.config_hash,
                "provenance_sha256": (
                    sha256_file(provenance_path) if provenance_path is not None else None
                ),
                "artifact_count": len(provenance.get("source_artifact_references", {})),
                "error_message": run.error_message,
            }
        )
    return records


def summarize_by_pipeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from faulttrace_reporting.stats import compute_clustered_bootstrap_ci

    summaries = []
    for pipeline_id in sorted({record["pipeline_id"] for record in records}):
        group = [record for record in records if record["pipeline_id"] == pipeline_id]
        scored = [record for record in group if record["is_correct"] is not None]
        certified = [record for record in group if record["policy_decision"] == "certified"]
        structural_certified = [
            record
            for record in group
            if record["structural_policy_counterfactual_decision"] == "certified"
        ]
        reason_counts = Counter(
            str(reason)
            for record in group
            for reason in record.get("certificate_reason_codes", [])
        )
        ratio_names = sorted(
            {
                name
                for record in group
                for name in record.get("certificate_ratios", {})
            }
        )
        mean_certificate_ratios = {
            name: sum(
                float(record["certificate_ratios"][name])
                for record in group
                if name in record.get("certificate_ratios", {})
            )
            / sum(name in record.get("certificate_ratios", {}) for record in group)
            for name in ratio_names
        }
        accuracy_ci = None
        if scored:
            _, accuracy_interval = compute_clustered_bootstrap_ci(
                [float(record["is_correct"] is True) for record in scored],
                [str(record["query_id"]) for record in scored],
                samples=10_000,
                seed=42,
            )
            accuracy_ci = list(accuracy_interval)
        summaries.append(
            {
                "pipeline_id": pipeline_id,
                "run_count": len(group),
                "completed_count": sum(record["status"] == "completed" for record in group),
                "accuracy": (
                    sum(record["is_correct"] is True for record in scored) / len(scored)
                    if scored
                    else None
                ),
                "accuracy_clustered_bootstrap_95_ci": accuracy_ci,
                "effective_query_count": len({record["query_id"] for record in scored}),
                "certificate_reason_counts": dict(sorted(reason_counts.items())),
                "mean_certificate_ratios": mean_certificate_ratios,
                "certified_rate": len(certified) / len(group) if group else None,
                "structural_policy_counterfactual_certified_rate": (
                    len(structural_certified) / len(group) if group else None
                ),
                "false_certification_rate": (
                    sum(record["is_correct"] is False for record in certified) / len(certified)
                    if certified
                    else None
                ),
                "structural_policy_counterfactual_false_certification_rate": (
                    sum(record["is_correct"] is False for record in structural_certified)
                    / len(structural_certified)
                    if structural_certified
                    else None
                ),
                "mean_latency_ms": (
                    sum(float(record["latency_ms"]) for record in group if record["latency_ms"] is not None)
                    / sum(record["latency_ms"] is not None for record in group)
                    if any(record["latency_ms"] is not None for record in group)
                    else None
                ),
            }
        )
    return summaries


def _load_typed_answer(artifact_refs: dict[str, str], fallback: Any) -> Any:
    answer_path_value = artifact_refs.get("aggregation_result")
    if answer_path_value:
        answer_path = Path(answer_path_value)
        if answer_path.exists():
            payload = json.loads(answer_path.read_text(encoding="utf-8"))
            if "answer" in payload:
                return payload["answer"]
            if "result" in payload:
                return payload["result"]
    try:
        return ast.literal_eval(str(fallback))
    except (SyntaxError, ValueError):
        return fallback


def run_attribution_audit(
    db: Any,
    records: list[dict[str, Any]],
    output_dir: Path,
    per_pipeline: int,
) -> dict[str, Any]:
    """Run a deterministic, query-balanced attribution audit on injected faults."""
    import pandas as pd
    from faulttrace_api.database import QueryRow, RunRow
    from faulttrace_core.models import GoldAnswer, PipelineRun, QuerySpec, RunStatus
    from faulttrace_pipelines.attribution import CounterfactualAttributor
    from faulttrace_reporting.diagnostics import (
        FAULT_LABELS,
        cross_fitted_shapley_multilabel_metrics,
        fault_localization_metrics,
    )
    from faulttrace_reporting.stats import compute_clustered_bootstrap_ci, paired_permutation_test

    selected = []
    for pipeline_id in FAULT_LABELS:
        candidates = sorted(
            (
                record
                for record in records
                if record["pipeline_id"] == pipeline_id and record["status"] == "completed"
            ),
            key=lambda record: (str(record["query_id"]), int(record["seed"])),
        )
        selected.extend(candidates[:per_pipeline])

    attributor = CounterfactualAttributor()
    attributor.lattice_runner.artifacts_dir = output_dir / "attribution_audit" / "lattice"
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in selected:
        run_row = db.query(RunRow).filter(RunRow.run_id == record["run_id"]).one()
        query_row = db.query(QueryRow).filter(QueryRow.query_id == record["query_id"]).one()
        try:
            query = QuerySpec.model_validate(json.loads(query_row.spec_json))
            gold = GoldAnswer.model_validate(json.loads(query_row.gold_json))
            artifact_refs = json.loads(run_row.artifact_refs_json or "{}")
            parent = PipelineRun(
                run_id=run_row.run_id,
                query_id=run_row.query_id,
                pipeline_id=run_row.pipeline_id,
                provider_id=run_row.provider_id,
                execution_seed=int(record["seed"]),
                status=RunStatus.COMPLETED,
                answer=_load_typed_answer(artifact_refs, run_row.answer),
                gold_answer_value=gold.answer_value,
                is_correct=run_row.is_correct,
                loss=run_row.loss,
                config_hash=run_row.config_hash or "",
                artifact_references=artifact_refs,
            )
            world_path = (
                Path(os.environ["FAULTTRACE_DATA_ROOT"])
                / "generated"
                / "worlds"
                / record["world_id"]
                / "records.parquet"
            )
            corpus = pd.read_parquet(world_path)
            result = attributor.attribute(parent, query, gold, corpus).to_dict()
            result["seed"] = record["seed"]
            result["true_faults"] = sorted(FAULT_LABELS[parent.pipeline_id])
            outputs.append(result)
        except Exception as exc:
            failures.append({"run_id": record["run_id"], "error": str(exc)})

    dominant_records = []
    outcome_active_records = []
    for item in outputs:
        dominant_item = dict(item)
        dominant_item.pop("predicted_faults", None)
        dominant_records.append(dominant_item)
        outcome_active_records.append(
            {**item, "predicted_faults": item.get("outcome_active_faults", [])}
        )

    artifact_metrics = fault_localization_metrics(outputs)
    dominant_metrics = fault_localization_metrics(dominant_records)
    outcome_active_metrics = fault_localization_metrics(outcome_active_records)
    cross_fitted_metrics = cross_fitted_shapley_multilabel_metrics(outputs)

    def exact_vector(items: list[dict[str, Any]], mode: str) -> list[float]:
        values = []
        for item in items:
            truth = set(FAULT_LABELS[item["pipeline_id"]])
            if mode == "artifact":
                prediction = set(item.get("artifact_discrepancy_faults", []))
            else:
                dominant = item.get("dominant_fault")
                prediction = set() if dominant in (None, "none", "unavailable") else {dominant}
            values.append(float(prediction == truth))
        return values

    artifact_exact = exact_vector(outputs, "artifact")
    dominant_exact = exact_vector(outputs, "dominant")
    if outputs:
        artifact_mean, artifact_ci = compute_clustered_bootstrap_ci(
            artifact_exact,
            [str(item["query_id"]) for item in outputs],
            samples=10_000,
            seed=47,
        )
        dominant_mean, dominant_ci = compute_clustered_bootstrap_ci(
            dominant_exact,
            [str(item["query_id"]) for item in outputs],
            samples=10_000,
            seed=47,
        )
        paired_exact_p = paired_permutation_test(
            artifact_exact, dominant_exact, samples=10_000, seed=47
        )
    else:
        artifact_mean = dominant_mean = 0.0
        artifact_ci = dominant_ci = (0.0, 0.0)
        paired_exact_p = 1.0

    diagnostics = {
        **artifact_metrics,
        "primary_diagnostic_target": "stage-matched oracle artifact discrepancy",
        "artifact_discrepancy_metrics": artifact_metrics,
        "outcome_active_shapley_metrics": outcome_active_metrics,
        "dominant_component_metrics": dominant_metrics,
        "cross_fitted_shapley_multilabel": cross_fitted_metrics,
        "paired_exact_set_comparison": {
            "artifact_mean": artifact_mean,
            "artifact_query_clustered_95_ci": list(artifact_ci),
            "dominant_mean": dominant_mean,
            "dominant_query_clustered_95_ci": list(dominant_ci),
            "paired_randomization_p_value": paired_exact_p,
            "samples": 10_000,
            "unit": "audited run, with query-clustered confidence intervals",
        },
    }
    diagnostics.update(
        {
            "requested_per_pipeline": per_pipeline,
            "selected_run_count": len(selected),
            "successful_lattice_count": len(outputs),
            "failed_lattice_count": len(failures),
            "mean_rea_normalized_loss": (
                sum(float(item["interventions"]["rea"]["normalized_loss"]) for item in outputs)
                / len(outputs)
                if outputs
                else None
            ),
            "mean_abs_efficiency_residual": (
                sum(abs(float(item["interaction_term"])) for item in outputs) / len(outputs)
                if outputs
                else None
            ),
            "selection": "first query_id/seed pairs per pipeline after deterministic sort",
        }
    )
    audit_dir = output_dir / "attribution_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "attributions.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (audit_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )
    (audit_dir / "metrics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts"
        / "verified"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--attribution-per-pipeline",
        type=int,
        default=20,
        help="Deterministic audit sample per injected-fault pipeline (0 disables).",
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse non-clean output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    configure_isolated_environment(output_dir)

    # Imports intentionally occur after isolation environment variables are set.
    import pandas as pd
    from faulttrace_api.database import get_session_factory, init_db
    from faulttrace_api.models import SeedDemoRequest
    from faulttrace_api.routes.demo import seed_demo
    from faulttrace_reporting import FigureGenerator, MetricsComputer, ReproducibilityBundle
    from faulttrace_reporting.experiments import ExperimentSpec, ResumableMatrixRunner

    init_db()
    db = get_session_factory()()
    started_at = datetime.now(UTC)
    try:
        seed_result = asyncio.run(
            seed_demo(
                SeedDemoRequest(seed=42, scales=[10, 50, 200, 1000], overwrite=True),
                db,
            )
        )
        data_root = output_dir / "data"
        before_inventory = dataset_inventory(data_root)
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "seed_result": seed_result,
                    "immutable_after_generation": True,
                    "files": before_inventory,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        make_dataset_read_only(data_root)

        experiment_results = []
        all_records: list[dict[str, Any]] = []
        for config_path in checked_in_configs():
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            config_payload["output_root"] = str((output_dir / "experiments").resolve())
            config_payload["cache_policy"] = "recompute"
            spec = ExperimentSpec.model_validate(config_payload)
            runner = ResumableMatrixRunner(spec, db)
            jobs = runner.expand_matrix()
            job_map = {job.job_id: job for job in jobs}
            plan = runner.dry_run()
            result = runner.run()
            records = collect_runs(db, runner.config_hash, job_map)
            all_records.extend(records)

            experiment_dir = output_dir / "experiments" / runner.config_hash
            metrics = MetricsComputer.compute_all(records)
            figure_paths = FigureGenerator(records, experiment_dir / "figures").generate_all()
            bundle_path = ReproducibilityBundle.export_bundle(
                runner.config_hash,
                spec.model_dump(mode="json"),
                pd.DataFrame(records),
                experiment_dir,
            )
            bundle_valid, bundle_errors = ReproducibilityBundle.verify_bundle(bundle_path)
            experiment_results.append(
                {
                    "name": spec.name,
                    "config_path": config_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "config_file_sha256": sha256_file(config_path),
                    "config_hash": runner.config_hash,
                    "plan": plan,
                    "execution": result,
                    "metrics": metrics.model_dump(mode="json"),
                    "figures": [str(path) for path in figure_paths],
                    "bundle_path": str(bundle_path),
                    "bundle_valid": bundle_valid,
                    "bundle_errors": bundle_errors,
                }
            )

        after_inventory = dataset_inventory(data_root)
        if before_inventory != after_inventory:
            raise RuntimeError("Immutable dataset fingerprint changed during experiment execution")

        attribution_audit = (
            run_attribution_audit(
                db, all_records, output_dir, arguments.attribution_per_pipeline
            )
            if arguments.attribution_per_pipeline > 0
            else {"measurement_status": "disabled"}
        )
        summary = {
            "schema_version": "1.0.0",
            "measurement_status": "measured_from_clean_local_execution",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "source_state": source_state(),
            "dataset_manifest_sha256": sha256_file(output_dir / "dataset_manifest.json"),
            "experiment_count": len(experiment_results),
            "run_count": len(all_records),
            "failed_run_count": sum(record["status"] != "completed" for record in all_records),
            "experiments": experiment_results,
            "pipeline_summary": summarize_by_pipeline(all_records),
            "attribution_audit": attribution_audit,
        }
        results_path = output_dir / "verified_results.json"
        results_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(json.dumps({"output_dir": str(output_dir), "results": str(results_path), **summary["source_state"]}))
        return 0 if summary["failed_run_count"] == 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
