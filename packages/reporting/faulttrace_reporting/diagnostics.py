"""Fault-localization and selective-risk metrics for publication reports."""

from __future__ import annotations

import hashlib
from typing import Any

FAULT_LABELS: dict[str, frozenset[str]] = {
    "P0-deterministic-scope-baseline": frozenset(),
    "P1-wrong-scope": frozenset({"scope"}),
    "P2-wrong-facts": frozenset({"facts"}),
    "P3-wrong-aggregation": frozenset({"aggregation"}),
    "P4-compound-scope-facts": frozenset({"scope", "facts"}),
    "P5-full-compound": frozenset({"scope", "facts", "aggregation"}),
}


def fault_localization_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate dominant and multilabel fault predictions against injected truth.

    Records require ``pipeline_id`` plus either ``dominant_fault`` or a
    ``predicted_faults`` iterable. Unknown pipelines are excluded and counted.
    """
    evaluable: list[tuple[set[str], set[str]]] = []
    excluded = 0
    for record in records:
        truth = FAULT_LABELS.get(str(record.get("pipeline_id")))
        if truth is None:
            excluded += 1
            continue
        if record.get("predicted_faults") is not None:
            predicted = {str(value) for value in record["predicted_faults"]}
        else:
            dominant = record.get("dominant_fault")
            predicted = set() if dominant in (None, "none", "unavailable") else {str(dominant)}
        evaluable.append((set(truth), predicted))

    labels = ("scope", "facts", "aggregation")
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in labels:
        tp = sum(label in truth and label in prediction for truth, prediction in evaluable)
        fp = sum(label not in truth and label in prediction for truth, prediction in evaluable)
        fn = sum(label in truth and label not in prediction for truth, prediction in evaluable)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    exact = sum(truth == prediction for truth, prediction in evaluable)
    top1 = sum(
        len(truth) == 0 and len(prediction) == 0
        or len(prediction) == 1 and next(iter(prediction)) in truth
        for truth, prediction in evaluable
    )
    micro_tp = sum(len(truth & prediction) for truth, prediction in evaluable)
    micro_fp = sum(len(prediction - truth) for truth, prediction in evaluable)
    micro_fn = sum(len(truth - prediction) for truth, prediction in evaluable)
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    example_f1 = sum(
        1.0
        if not truth and not prediction
        else 2 * len(truth & prediction) / (len(truth) + len(prediction))
        if truth or prediction
        else 0.0
        for truth, prediction in evaluable
    )
    differing_labels = sum(len(truth ^ prediction) for truth, prediction in evaluable)
    return {
        "sample_count": len(evaluable),
        "excluded_unknown_pipeline_count": excluded,
        "exact_set_accuracy": exact / len(evaluable) if evaluable else 0.0,
        "top1_localization_accuracy": top1 / len(evaluable) if evaluable else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "example_f1": example_f1 / len(evaluable) if evaluable else 0.0,
        "hamming_loss": differing_labels / (len(evaluable) * len(labels)) if evaluable else 0.0,
        "mean_true_cardinality": (
            sum(len(truth) for truth, _ in evaluable) / len(evaluable) if evaluable else 0.0
        ),
        "mean_predicted_cardinality": (
            sum(len(prediction) for _, prediction in evaluable) / len(evaluable)
            if evaluable
            else 0.0
        ),
        "per_label": per_label,
        "ground_truth_source": "pipeline fault-injection definition",
    }


def _threshold_predictions(record: dict[str, Any], threshold: float) -> list[str]:
    return [
        str(component["component"])
        for component in record.get("components", [])
        if abs(float(component.get("shapley_value", 0.0))) > threshold
    ]


def cross_fitted_shapley_multilabel_metrics(
    records: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.025, 0.05, 0.1),
    folds: int = 2,
) -> dict[str, Any]:
    """Calibrate a Shapley multilabel threshold without scoring on its training fold.

    Query IDs, rather than rows, are deterministically assigned to folds. For
    each held-out fold, the threshold maximizing calibration-fold macro-F1 is
    selected; ties prefer exact-set accuracy and then the larger threshold.
    Predictions from held-out folds are pooled for the reported metrics.
    """
    if folds < 2:
        raise ValueError("folds must be at least two")
    if not records:
        return {"metrics": fault_localization_metrics([]), "folds": []}

    assignments: list[int] = []
    for index, record in enumerate(records):
        cluster = str(record.get("query_id", f"row-{index}"))
        digest = hashlib.sha256(cluster.encode("utf-8")).digest()
        assignments.append(int.from_bytes(digest[:8], "big") % folds)

    cross_fitted: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for held_out in range(folds):
        calibration = [record for record, fold in zip(records, assignments, strict=True) if fold != held_out]
        evaluation = [record for record, fold in zip(records, assignments, strict=True) if fold == held_out]
        if not calibration or not evaluation:
            continue
        candidates: list[tuple[float, dict[str, Any]]] = []
        for threshold in thresholds:
            labeled = [
                {**record, "predicted_faults": _threshold_predictions(record, threshold)}
                for record in calibration
            ]
            candidates.append((threshold, fault_localization_metrics(labeled)))
        selected_threshold, calibration_metrics = max(
            candidates,
            key=lambda item: (
                float(item[1]["macro_f1"]),
                float(item[1]["exact_set_accuracy"]),
                item[0],
            ),
        )
        held_out_records = [
            {**record, "predicted_faults": _threshold_predictions(record, selected_threshold)}
            for record in evaluation
        ]
        cross_fitted.extend(held_out_records)
        fold_summaries.append(
            {
                "held_out_fold": held_out,
                "selected_threshold": selected_threshold,
                "calibration_sample_count": len(calibration),
                "evaluation_sample_count": len(evaluation),
                "calibration_macro_f1": calibration_metrics["macro_f1"],
            }
        )

    return {
        "metrics": fault_localization_metrics(cross_fitted),
        "folds": fold_summaries,
        "candidate_thresholds": list(thresholds),
        "protocol": "deterministic query-clustered cross-fitting; no row is scored by its training fold",
        "prediction_target": "outcome-active injected components, approximated by injected labels",
    }


def area_under_risk_coverage_curve(points: list[dict[str, Any]]) -> float | None:
    """Trapezoidal AURC for genuinely measured risk/coverage operating points."""
    measured = sorted(
        (
            (float(point["coverage_rate"]), float(point["risk"]))
            for point in points
            if point.get("measurement_status") == "measured"
            and point.get("coverage_rate") is not None
            and point.get("risk") is not None
        ),
        key=lambda item: item[0],
    )
    if len(measured) < 2:
        return None
    area = 0.0
    for (coverage_a, risk_a), (coverage_b, risk_b) in zip(
        measured, measured[1:], strict=False
    ):
        area += (coverage_b - coverage_a) * (risk_a + risk_b) / 2.0
    return area
