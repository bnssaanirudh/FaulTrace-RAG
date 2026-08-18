# Methodology Note — FaultTrace-RAG Attribution

## Overview

This document precisely defines the counterfactual attribution methodology used in FaultTrace-RAG. It describes the value function, Shapley computation, loss metrics, and the new text-benchmark extension.

---

## 1. Oracle-Lattice Attribution Framework

FaultTrace-RAG decomposes a RAG pipeline error into contributions from three components:

| Component | Symbol | Description |
|---|---|---|
| **Retrieval** | R | Which documents are retrieved (scope) |
| **Extraction** | E | Which facts are extracted from those documents |
| **Aggregation** | A | How facts are reduced to a final answer |

Each component can be **replaced with a deterministic oracle** that produces a perfect output for that stage. The 8-subset lattice (∅, {R}, {E}, {A}, {R,E}, {R,A}, {E,A}, {R,E,A}) defines all possible oracle replacement combinations.

---

## 2. Value Function

The value function `v(S)` measures how much error is recovered when the components in subset S are replaced with oracles:

```
v(S) = baseline_loss - loss(S)
```

Where:
- `baseline_loss` = normalized loss of the unmodified pipeline (subset ∅)
- `loss(S)` = normalized loss when components in S use oracle outputs

> [!IMPORTANT]
> **v(S) can be negative.** This occurs when an oracle replacement makes the answer *worse* — for example, if the pipeline's retrieval was accidentally correct but oracle extraction changes the fact set and degrades the final answer. Negative values are preserved and reported; they are NOT clamped to zero.

---

## 3. Loss Functions

| Query Family | Loss Metric | Formula |
|---|---|---|
| Count, Sum, Mean, Proportion, Comparison | Normalized absolute error | `min(|p - g| / max(|g|, 1.0), 1.0)` |
| TopK | Jaccard distance | `1 - |P ∩ G| / |P ∪ G|` |
| Trend | L1 distance over time buckets | `min(Σ|g_k - p_k| / max(Σ|g_k|, 1.0), 1.0)` |
| Text (SupportStatus) | Status-to-loss mapping | See table below |

**Support Status Loss Mapping:**

| Status | Loss |
|---|---|
| `supported` | 0.0 |
| `partially_supported` | 0.5 |
| `insufficient_evidence` | 0.5 |
| `conflicting` | 0.8 |
| `unsupported` | 1.0 |

---

## 4. Shapley Value Computation

For three components {R, E, A}, the exact Shapley value is:

```
φ_R = (1/3)(v({R}) - v(∅))
    + (1/6)(v({R,E}) - v({E}))
    + (1/6)(v({R,A}) - v({A}))
    + (1/3)(v({R,E,A}) - v({E,A}))
```

Weights are derived from the uniform distribution over orderings of 3 elements:
- |S|=0: weight 1/3 (1 subset of size 0)
- |S|=1: weight 1/6 each (2 subsets of size 1 not containing i)
- |S|=2: weight 1/3 (1 subset of size 2 not containing i)

> [!IMPORTANT]
> **Shapley values are NOT clamped to [0, 1].** They can be negative (harmful oracle) or exceed 1.0 in theory (though this is rare with our normalized loss). The clamping that existed in prior versions was mathematically incorrect and has been removed.

---

## 5. Interaction Term

The interaction term captures error that is not attributable to any single component:

```
interaction = v(REA) - (φ_R + φ_E + φ_A)
```

**Efficiency Axiom (guaranteed):**
```
φ_R + φ_E + φ_A + interaction = v(REA) = total_recoverable_error
```

The interaction term:
- **Positive**: Sub-additive effect — components independently recover more than they do together
- **Negative**: Super-additive effect — components are complementary (fixing one alone helps more than fixing two together separately)
- **Zero**: All error is cleanly decomposable into individual component contributions

---

## 6. Total Recoverable Error

```
total_recoverable_error = v(REA) = baseline_loss - loss(REA)
```

This represents the maximum error that *could* be recovered by perfect oracle replacements. If `total_recoverable_error < baseline_loss`, some error is **not recoverable** even with all oracle components — this indicates systematic bias in the query spec or gold answer.

---

## 7. Dominant Fault

The dominant fault is determined by the highest **absolute** Shapley value:

```python
dominant = max({"scope": φ_R, "facts": φ_E, "aggregation": φ_A}, key=abs)
```

Absolute value is used because a strongly negative phi (harmful oracle) is just as informative as a strongly positive phi.

---

## 8. Invalid Interventions

When an intervention cannot be evaluated (e.g. missing artifact, missing parent extraction), the subset returns `status="invalid"` and `v(S) = 0.0`. Invalid interventions do NOT generate a negative Shapley value — they contribute nothing to the attribution. The `natural_language_summary` field records the specific reason for invalidity.

---

## 9. Text Attribution Extension

`TextAttributor` in `faulttrace_pipelines.text_attribution` applies the same 8-subset lattice to text-benchmark pipelines where the value function is based on `SupportStatus` rather than numeric error.

Components in text benchmarks:
- **R**: Which documents are retrieved
- **E**: Which facts/spans are extracted from those documents
- **A**: How facts are assembled into the final `SupportStatus` assessment

---

## 10. Known Limitations

1. The oracle assumes that oracle-produced outputs at stage N are compatible inputs for stage N+1. If the oracle scope returns records that the pipeline's extraction model has never seen, the E-stage may produce systematically different outputs than expected. This is inherent to the 3-component counterfactual model.

2. Shapley computation requires all 8 subset interventions to be valid. Missing artifacts cause those subsets to return `v(S) = 0.0`, which may under-attribute error.

3. For text benchmarks, the `SupportStatus` loss mapping is a heuristic. It does not reflect any learned utility function. Two different queries with `SupportStatus.PARTIALLY_SUPPORTED` are treated as equal loss regardless of how much evidence was present.

4. The deterministic fixture extractor is not a trained model and produces rule-based results. Reported `support_status` from the fixture extractor reflects keyword overlap, not semantic understanding.
