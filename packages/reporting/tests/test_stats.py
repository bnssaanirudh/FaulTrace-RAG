"""
Unit tests for metrics and bootstrap statistical functions.
"""

from __future__ import annotations

import pytest
from faulttrace_reporting.diagnostics import (
    area_under_risk_coverage_curve,
    cross_fitted_shapley_multilabel_metrics,
    fault_localization_metrics,
)
from faulttrace_reporting.metrics import MetricsComputer
from faulttrace_reporting.stats import (
    compute_clustered_bootstrap_ci,
    compute_paired_bootstrap_ci,
    holm_adjusted_pvalues,
    holm_bonferroni_correction,
    paired_permutation_test,
)


def test_paired_bootstrap_equal():
    """Test bootstrap CI for identical sample arrays."""
    data1 = [1.0, 0.0, 1.0, 1.0, 0.0]
    data2 = [1.0, 0.0, 1.0, 1.0, 0.0]

    mean_diff, (lower, upper), cohens_d = compute_paired_bootstrap_ci(
        data1, data2, confidence_level=0.95, samples=100
    )

    assert mean_diff == 0.0
    assert lower == 0.0
    assert upper == 0.0
    assert cohens_d == 0.0


def test_paired_bootstrap_difference():
    """Test bootstrap CI for different sample arrays."""
    data1 = [1.0, 1.0, 1.0, 1.0, 1.0]
    data2 = [0.0, 0.0, 0.0, 0.0, 0.0]

    mean_diff, (lower, upper), cohens_d = compute_paired_bootstrap_ci(
        data1, data2, confidence_level=0.95, samples=100
    )

    assert mean_diff == 1.0
    assert lower == 1.0
    assert upper == 1.0
    assert cohens_d > 0.0


def test_holm_bonferroni_adjustment():
    """Test Holm-Bonferroni step-down correction logic."""
    p_values = [0.005, 0.012, 0.045, 0.230]
    # Rankings sorted:
    # 0.005 (rank 0, threshold = 0.05 / 4 = 0.0125) -> True
    # 0.012 (rank 1, threshold = 0.05 / 3 = 0.0166) -> True
    # 0.045 (rank 2, threshold = 0.05 / 2 = 0.025)  -> False (stops)
    # 0.230 -> False

    rejections = holm_bonferroni_correction(p_values, alpha=0.05)
    assert rejections == [True, True, False, False]


def test_clustered_bootstrap_is_deterministic_and_query_clustered():
    mean, interval = compute_clustered_bootstrap_ci(
        [1.0, 1.0, 0.0, 0.0], ["q1", "q1", "q2", "q2"], samples=500, seed=7
    )
    assert mean == 0.5
    assert interval == (0.0, 1.0)


def test_paired_permutation_and_holm_adjusted_values():
    assert paired_permutation_test([1, 1, 1], [1, 1, 1], samples=100) == 1.0
    adjusted = holm_adjusted_pvalues([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_fault_localization_metrics_support_compound_faults():
    result = fault_localization_metrics(
        [
            {"pipeline_id": "P1-wrong-scope", "dominant_fault": "scope"},
            {
                "pipeline_id": "P4-compound-scope-facts",
                "predicted_faults": ["scope", "facts"],
            },
            {"pipeline_id": "P3-wrong-aggregation", "dominant_fault": "facts"},
        ]
    )
    assert result["sample_count"] == 3
    assert result["exact_set_accuracy"] == pytest.approx(2 / 3)
    assert result["top1_localization_accuracy"] == pytest.approx(1 / 3)
    assert result["micro_f1"] == pytest.approx(0.75)
    assert result["hamming_loss"] == pytest.approx(2 / 9)


def test_cross_fitted_multilabel_threshold_never_scores_training_rows():
    records = []
    for index in range(12):
        pipeline_id = "P1-wrong-scope" if index % 2 else "P0-deterministic-scope-baseline"
        score = 0.8 if index % 2 else 0.0
        records.append(
            {
                "query_id": f"q-{index}",
                "pipeline_id": pipeline_id,
                "components": [
                    {"component": "scope", "shapley_value": score},
                    {"component": "facts", "shapley_value": 0.0},
                    {"component": "aggregation", "shapley_value": 0.0},
                ],
            }
        )
    result = cross_fitted_shapley_multilabel_metrics(records)
    assert sum(fold["evaluation_sample_count"] for fold in result["folds"]) == len(records)
    assert result["metrics"]["exact_set_accuracy"] == 1.0
    assert "no row is scored by its training fold" in result["protocol"]


def test_aurc_uses_only_measured_points():
    area = area_under_risk_coverage_curve(
        [
            {"coverage_rate": 0.5, "risk": 0.1, "measurement_status": "measured"},
            {"coverage_rate": 1.0, "risk": 0.3, "measurement_status": "measured"},
            {"coverage_rate": 0.7, "risk": 99.0, "measurement_status": "simulated"},
        ]
    )
    assert area == pytest.approx(0.1)


def test_metrics_computer_empty():
    """Test metrics calculation handles empty runs cleanly."""
    res = MetricsComputer.compute_all([])
    assert res.sample_count == 0
    assert res.accuracy == 0.0
    assert res.selective_risk == 0.0


def test_metrics_computer_completed():
    """Test metrics calculator over standard completed pipeline runs."""
    runs = [
        {
            "status": "completed",
            "is_correct": True,
            "loss": 0.0,
            "latency_ms": 10.0,
            "pipeline_id": "P0-deterministic-scope-baseline",
            "policy_decision": "certified",
        },
        {
            "status": "completed",
            "is_correct": False,
            "loss": 1.0,
            "latency_ms": 20.0,
            "pipeline_id": "P1-wrong-scope",
            "policy_decision": "abstain",
        },
        {
            "status": "completed",
            "is_correct": True,
            "loss": 0.0,
            "latency_ms": 30.0,
            "pipeline_id": "P0-deterministic-scope-baseline",
            "policy_decision": "certified",
        },
    ]

    res = MetricsComputer.compute_all(runs)
    assert res.sample_count == 3
    assert res.failure_count == 0
    assert res.accuracy == pytest.approx(2 / 3)
    assert res.mean_loss == pytest.approx(1 / 3)
    assert res.certified_rate == pytest.approx(2 / 3)
    assert res.selective_risk == 0.0
    assert res.false_certification_rate == 0.0
    assert res.mean_latency_ms == pytest.approx(20.0)
