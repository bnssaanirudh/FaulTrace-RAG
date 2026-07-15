# Loss Functions for Fault Attribution

This document describes the typed loss functions utilized in the FaultTrace-RAG counterfactual attribution engine. Because different aggregate answers have different topological properties (e.g. discrete top-K sets vs. continuous means), we must apply family-specific metrics to properly evaluate pipeline subset predictions against the Gold Truth.

## Normalization and Shape
All loss functions return a unified `normalized_loss` scalar in the range `[0.0, 1.0]`. 
This normalized property is mathematically necessary to combine and distribute subset errors via exact Shapley values.

A `LossDiagnostic` is recorded for each prediction, containing:
1. `normalized_loss`: The unified scalar in [0.0, 1.0]
2. `raw_error`: The unnormalized absolute difference (for continuous types)
3. `jaccard_distance`: The set distance metric (for set types)
4. `status`: The evaluation state (`valid`, `invalid`, `abstained`, `partial`)

## Supported Families

### 1. Scalar Aggregates (Count, Sum, Mean, Proportion, Comparison)
These queries return a single continuous or discrete numeric value.
- **Raw Metric**: Absolute error `|P - G|`
- **Normalizer**: `max(|G|, 1.0)`
- **Unified Metric**: `min(|P - G| / max(|G|, 1.0), 1.0)`
- **Failure Status**: If the prediction is a non-numeric string or object, it is evaluated as `status = "invalid"` and yields `1.0` normalized loss.

### 2. Set Aggregates (Top-K)
These queries return an ordered or unordered set of grouped values. The evaluation is rank-aware/set-aware.
- **Metric**: Jaccard Distance.
- Let `Set(P)` and `Set(G)` be the sets of keys extracted from the prediction and gold truths respectively.
- `jaccard_similarity = |Set(P) ∩ Set(G)| / |Set(P) ∪ Set(G)|`
- **Unified Metric**: `1.0 - jaccard_similarity`

### 3. Time Series / Histograms (Trend)
These queries return lists of `{ bucket: "YYYY-MM", value: X }`.
- **Metric**: Normalized L1 Distance over all observed buckets.
- **Normalizer**: The sum of absolute values in the Gold truth `sum(|G_i|)`.
- Let `K` be the union of bucket keys in prediction and gold.
- `L1_error = sum(|P_k - G_k| for k in K)`
- **Unified Metric**: `min(L1_error / max(sum(|G_i|), 1.0), 1.0)`

### 4. Edge Cases
- **Abstained**: If the pipeline yields `None` but the gold is not `None`, it is marked as `abstained` with loss `1.0`.
- **Zero Gold Cases**: If `G=0` and `P=0`, the loss is `0.0`. If `G=0` and `P>0`, the error is clamped to `1.0`. 
- **Exact Matches**: If the types fallback and a tolerance equivalence passes, loss is exactly `0.0`.
