"""
Statistics engine: bootstrap confidence intervals, paired differences, and multiple-comparison adjustments.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def compute_clustered_bootstrap_ci(
    values: list[float],
    cluster_ids: list[str],
    confidence_level: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap a mean by resampling independent clusters, not repeated rows.

    This is the appropriate default when multiple seeds, models, or pipelines
    reuse the same query. All observations for a sampled query move together.
    """
    if not values or len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must be non-empty and have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster_id in zip(values, cluster_ids, strict=True):
        grouped[str(cluster_id)].append(float(value))
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        rows = [value for cluster in sampled for value in grouped[str(cluster)]]
        boot_means[index] = float(np.mean(rows))
    alpha = 1.0 - confidence_level
    interval = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(values)), (float(interval[0]), float(interval[1]))


def paired_permutation_test(
    data1: list[float],
    data2: list[float],
    samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Return a two-sided paired randomization-test p-value."""
    if not data1 or len(data1) != len(data2):
        raise ValueError("paired samples must be non-empty and have equal length")
    diffs = np.asarray(data1, dtype=float) - np.asarray(data2, dtype=float)
    observed = abs(float(np.mean(diffs)))
    if np.allclose(diffs, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=len(diffs))
        extreme += int(abs(float(np.mean(diffs * signs))) >= observed - 1e-15)
    return (extreme + 1) / (samples + 1)


def holm_adjusted_pvalues(p_values: list[float]) -> list[float]:
    """Return monotone Holm-adjusted p-values in the original order."""
    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def compute_paired_bootstrap_ci(
    data1: list[float],
    data2: list[float],
    confidence_level: float = 0.95,
    samples: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float], float]:
    """
    Computes paired difference bootstrap confidence intervals between two sample arrays.
    Returns: (mean_difference, (lower_bound, upper_bound), cohens_d_effect_size)
    """
    random.seed(seed)
    np.random.seed(seed)

    if not data1 or not data2 or len(data1) != len(data2):
        return 0.0, (0.0, 0.0), 0.0

    n = len(data1)
    diffs = np.array(data1) - np.array(data2)
    mean_diff = float(np.mean(diffs))

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(samples):
        resampled_indices = np.random.randint(0, n, size=n)
        resampled_diffs = diffs[resampled_indices]
        bootstrap_means.append(np.mean(resampled_diffs))

    # Percentiles
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    lower = float(np.percentile(bootstrap_means, lower_pct))
    upper = float(np.percentile(bootstrap_means, upper_pct))

    # Cohen's d effect size
    std_dev = np.std(diffs, ddof=1) if n > 1 else 1.0
    if std_dev == 0:
        std_dev = 1.0
    cohens_d = mean_diff / std_dev

    return mean_diff, (lower, upper), float(cohens_d)


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Applies Holm-Bonferroni correction to a list of p-values to control Family-Wise Error Rate (FWER).
    Returns list of booleans: True if the hypothesis is rejected (statistically significant), False otherwise.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    sorted_indices = sorted(range(m), key=lambda k: p_values[k])
    rejections = [False] * m

    for rank, idx in enumerate(sorted_indices):
        p_val = p_values[idx]
        # Holm threshold: alpha / (m - rank)
        threshold = alpha / (m - rank)
        if p_val < threshold:
            rejections[idx] = True
        else:
            # Stop: subsequent larger p-values are not significant
            break

    return rejections
