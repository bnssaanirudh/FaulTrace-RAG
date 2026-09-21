"""Derive traceable subgroup analyses from a verified FaultTrace result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    REPOSITORY_ROOT / "packages" / "reporting",
    REPOSITORY_ROOT / "packages" / "pipelines",
):
    sys.path.insert(0, str(source_root))

from faulttrace_reporting.diagnostics import FAULT_LABELS, fault_localization_metrics
from faulttrace_reporting.stats import compute_clustered_bootstrap_ci


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(record: dict[str, Any], mode: str) -> list[str]:
    if mode == "artifact_discrepancy":
        return [str(value) for value in record.get("artifact_discrepancy_faults", [])]
    if mode == "outcome_active":
        return [str(value) for value in record.get("outcome_active_faults", [])]
    dominant = record.get("dominant_fault")
    return [] if dominant in (None, "none", "unavailable") else [str(dominant)]


def subgroup(records: list[dict[str, Any]], fault_count: int, mode: str) -> dict[str, Any]:
    chosen = [
        {**record, "predicted_faults": _prediction(record, mode)}
        for record in records
        if len(FAULT_LABELS.get(str(record.get("pipeline_id")), frozenset())) == fault_count
    ]
    metrics = fault_localization_metrics(chosen)
    if chosen:
        exact = [
            float(set(record["predicted_faults"]) == set(FAULT_LABELS[record["pipeline_id"]]))
            for record in chosen
        ]
        mean, interval = compute_clustered_bootstrap_ci(
            exact,
            [str(record["query_id"]) for record in chosen],
            samples=10_000,
            seed=42,
        )
        metrics["exact_set_query_clustered_mean"] = mean
        metrics["exact_set_query_clustered_bootstrap_95_ci"] = list(interval)
    metrics["prediction_mode"] = mode
    return metrics


def all_subgroups(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "clean_control": subgroup(records, 0, mode),
        "single_fault": subgroup(records, 1, mode),
        "compound_fault": {
            "two_fault": subgroup(records, 2, mode),
            "three_fault": subgroup(records, 3, mode),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verified_directory", type=Path)
    parser.add_argument("--allow-fast-mode", action="store_true", help="Allow processing FAST_MODE exploratory artifacts")
    args = parser.parse_args()
    root = args.verified_directory.resolve()
    results_path = root / "verified_results.json"
    attribution_path = root / "attribution_audit" / "attributions.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    
    if not args.allow_fast_mode:
        if results.get("configuration", {}).get("FAST_MODE", False):
            raise ValueError("Refusing analysis: Source artifact was generated in FAST_MODE. Use --allow-fast-mode to override.")
            
    attributions = json.loads(attribution_path.read_text(encoding="utf-8"))
    if results.get("failed_run_count") != 0 or results["attribution_audit"].get(
        "failed_lattice_count"
    ) != 0:
        raise ValueError("Refusing analysis because the verified execution has failures")

    rea_losses = [
        float(record["interventions"]["rea"]["normalized_loss"])
        for record in attributions
    ]
    analysis = {
        "schema_version": "2.0.0",
        "input_results_sha256": sha256_file(results_path),
        "input_attributions_sha256": sha256_file(attribution_path),
        "sample_count": len(attributions),
        "diagnostic_subgroups": {
            "artifact_discrepancy": all_subgroups(attributions, "artifact_discrepancy"),
            "outcome_active": all_subgroups(attributions, "outcome_active"),
            "dominant_component": all_subgroups(attributions, "dominant_component"),
        },
        "full_oracle_zero_loss_rate": (
            sum(loss == 0.0 for loss in rea_losses) / len(rea_losses) if rea_losses else None
        ),
        "mean_full_oracle_loss": (
            sum(rea_losses) / len(rea_losses) if rea_losses else None
        ),
        "analysis_scope": (
            "Deterministic balanced audit only; not external-model or external-dataset evidence."
        ),
    }
    output_path = root / "secondary_analysis.json"
    output_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), **analysis}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
