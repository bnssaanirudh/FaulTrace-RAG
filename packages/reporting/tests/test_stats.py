"""
Unit tests for metrics and bootstrap statistical functions.
"""

from __future__ import annotations

import pytest
from faulttrace_reporting.metrics import MetricsComputer
from faulttrace_reporting.stats import compute_paired_bootstrap_ci, holm_bonferroni_correction


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
