"""
Statistics engine: bootstrap confidence intervals, paired differences, and multiple-comparison adjustments.
"""

from __future__ import annotations

import random

import numpy as np


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
