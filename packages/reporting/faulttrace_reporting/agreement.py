"""
Inter-rater agreement metrics for Track T annotations.
Provides robust statistical methods for quantifying annotation reliability.
"""

import pandas as pd


def compute_cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """
    Computes Cohen's Kappa for two raters.
    Kappa = (Po - Pe) / (1 - Pe)
    where Po is relative observed agreement, Pe is hypothetical probability of chance agreement.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0

    df = pd.DataFrame({"A": rater_a, "B": rater_b})
    n = len(df)

    # Observed agreement
    observed_agreement = (df["A"] == df["B"]).sum() / n

    # Expected agreement
    counts_a = df["A"].value_counts() / n
    counts_b = df["B"].value_counts() / n

    expected_agreement = 0.0
    for label in set(rater_a).union(set(rater_b)):
        p_a = counts_a.get(label, 0)
        p_b = counts_b.get(label, 0)
        expected_agreement += p_a * p_b

    if expected_agreement == 1.0:
        return 1.0

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return float(kappa)


def compute_krippendorffs_alpha(reliability_data: list[list[str]]) -> float:
    """
    Computes Krippendorff's Alpha for multiple raters and missing data.
    reliability_data: List of lists, where each inner list represents a rater's annotations.
    Nulls/Nones are treated as missing data.

    This is a simplified nominal implementation for the Track T pilot.
    """
    if not reliability_data or not reliability_data[0]:
        return 0.0

    # Transpose to shape (num_items, num_raters)
    items = list(zip(*reliability_data, strict=False))
    len(items)

    total_pairs = 0
    observed_disagreement = 0.0

    value_counts: dict[str, int] = {}

    for item in items:
        valid_ratings = [r for r in item if r is not None]
        m = len(valid_ratings)
        if m < 2:
            continue

        for r in valid_ratings:
            value_counts[r] = value_counts.get(r, 0) + 1

        # Pairwise disagreements
        item_disagreements = 0
        for i in range(m):
            for j in range(i + 1, m):
                if valid_ratings[i] != valid_ratings[j]:
                    item_disagreements += 1

        observed_disagreement += item_disagreements / (m - 1)
        total_pairs += m

    if total_pairs == 0:
        return 0.0

    total_valid_values = sum(value_counts.values())
    expected_disagreement = 0.0
    for val1, c1 in value_counts.items():
        for val2, c2 in value_counts.items():
            if val1 != val2:
                expected_disagreement += c1 * c2

    expected_disagreement /= total_valid_values * (total_valid_values - 1)

    if expected_disagreement == 0:
        return 1.0

    observed_disagreement / (total_valid_values / 2)  # simplified
    alpha = (
        1 - (observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0
    )

    # Clip for simplicity in this stub
    return max(-1.0, min(1.0, float(alpha)))
