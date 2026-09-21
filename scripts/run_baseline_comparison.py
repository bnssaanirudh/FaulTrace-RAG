"""Baseline comparison evaluation runner.

Usage:
    python scripts/run_baseline_comparison.py \\
        --cases research/canonical/annotations/adjudicated.jsonl \\
        --baseline ragchecker \\
        --output paper/generated/table_baseline_comparison.csv

This script:
1. Loads adjudicated human annotation cases.
2. Runs the selected baseline adapter on each case.
3. Computes exact set accuracy, macro F1, micro F1, per-stage P/R/F1.
4. Computes bootstrap 95% CIs.
5. Outputs a CSV comparison table and JSON metrics file.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from faulttrace_pipelines.baselines import (
    BaselineAdapter,
    DiagnosisInput,
)


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def exact_set_accuracy(preds: list[set], golds: list[set]) -> float:
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def compute_f1_metrics(
    preds: list[set], golds: list[set], labels: list[str]
) -> dict[str, float]:
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(label in p and label in g for p, g in zip(preds, golds))
        fp = sum(label in p and label not in g for p, g in zip(preds, golds))
        fn = sum(label not in p and label in g for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)

    tp_all = sum(len(p & g) for p, g in zip(preds, golds))
    fp_all = sum(len(p - g) for p, g in zip(preds, golds))
    fn_all = sum(len(g - p) for p, g in zip(preds, golds))
    micro_prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    micro_rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "per_label": per_label,
    }


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_bootstrap)
    )
    lo = int(n_bootstrap * alpha / 2)
    hi = int(n_bootstrap * (1 - alpha / 2))
    return means[lo], means[hi]


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline comparison runner")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=["ragchecker", "ragas"],
        required=True,
        help="Which baseline adapter to run",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("paper/generated/table_baseline_comparison.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("paper/generated/baseline_metrics.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.baseline == "ragchecker":
        from faulttrace_pipelines.baselines.ragchecker_adapter import RAGCheckerAdapter
        adapter: BaselineAdapter = RAGCheckerAdapter(threshold=args.threshold)
    else:
        from faulttrace_pipelines.baselines.ragas_adapter import RAGASAdapter
        adapter = RAGASAdapter(threshold=args.threshold)

    raw_cases = load_cases(args.cases)
    diagnosis_inputs = [
        DiagnosisInput(
            case_id=c["case_id"],
            query=c["query"],
            retrieved_scope=c.get("retrieved_scope", []),
            extracted_facts=c.get("extracted_facts", []),
            predicted_answer=c["predicted_answer"],
            gold_answer=c["gold_answer"],
            error_magnitude=c.get("error_magnitude", 1.0),
            model=c.get("model", "unknown"),
            dataset=c.get("dataset", "unknown"),
        )
        for c in raw_cases
    ]

    outputs = adapter.diagnose_batch(diagnosis_inputs)
    labels = ["R", "E", "A", "G"]
    preds = [set(o.predicted_faults) - {"COMPOUND"} for o in outputs]
    golds = [set(c.get("adjudicated_faults", [])) for c in raw_cases]

    metrics = compute_f1_metrics(preds, golds, labels)
    acc = exact_set_accuracy(preds, golds)
    acc_ci = bootstrap_ci([float(p == g) for p, g in zip(preds, golds)])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "ExactSetAcc", "MacroF1", "MicroF1", "CI_lo", "CI_hi"])
        writer.writeheader()
        writer.writerow({
            "Method": adapter.name,
            "ExactSetAcc": f"{acc:.3f}",
            "MacroF1": f"{metrics['macro_f1']:.3f}",
            "MicroF1": f"{metrics['micro_f1']:.3f}",
            "CI_lo": f"{acc_ci[0]:.3f}",
            "CI_hi": f"{acc_ci[1]:.3f}",
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": adapter.name, "exact_set_accuracy": acc, **metrics}, f, indent=2)

    print(f"Baseline: {adapter.name}")
    print(f"Exact Set Accuracy: {acc:.3f} (95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}])")
    print(f"Macro F1: {metrics['macro_f1']:.3f}  Micro F1: {metrics['micro_f1']:.3f}")
    return 0


if __name__ == "__main__":
    exit(main())
