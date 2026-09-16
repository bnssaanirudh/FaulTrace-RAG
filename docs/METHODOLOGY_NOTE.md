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

## 5. Efficiency Residual

For an exact Shapley decomposition over the complete lattice, efficiency requires:

```
interaction = v(REA) - (φ_R + φ_E + φ_A)
```

**Efficiency Axiom (guaranteed):**
```
φ_R + φ_E + φ_A + interaction = v(REA) = total_recoverable_error
```

The retained `interaction` field is a backward-compatible numerical residual. It should be
zero up to floating-point tolerance for every complete valid lattice and must not be
interpreted as a separately identified higher-order interaction. Pairwise or higher-order
interactions require an explicit interaction index and are not estimated here.

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

When an intervention cannot be evaluated (for example, because a required parent artifact
is absent), the subset returns `status="invalid"`. The attribution engine rejects the
entire lattice and reports the reasons. It does not replace invalid interventions with
zeros, because doing so would silently bias the Shapley values.

---

## 9. Text Attribution Extension

`TextAttributor` in `faulttrace_pipelines.text_attribution` applies the same 8-subset lattice to text-benchmark pipelines where the value function is based on `SupportStatus` rather than numeric error.

Components in text benchmarks:
- **R**: Which documents are retrieved
- **E**: Which facts/spans are extracted from those documents
- **A**: How facts are assembled into the final `SupportStatus` assessment

The external SciFact audit instantiates this lattice with deterministic providers. Its
injected-fault labels are generated independently of the attribution calculation, and
all eight interventions are executed before Shapley scoring. This validates lattice
execution and fault-localization mechanics; it is not evidence that a learned extractor
or an LLM can recover the same interventions on unrestricted text.

---

## 10. External Text Metrics and Certificate Calibration

The deterministic external audit uses task-specific metrics rather than treating every
text dataset as the same prediction problem:

| Dataset | Prediction surface | Primary metrics |
|---|---|---|
| SciFact | Claim support status and evidence retrieval | Accuracy, macro F1, evidence recall@5 |
| HotpotQA distractor | Extracted evidence sentence | Normalized exact match, token F1, supporting-document recall@2 |
| RAGBench COVID-QA | Lexical/numeric source consistency | Coverage, false-certification rate, precision, recall |

For RAGBench, the score for a response is the minimum content-token coverage of any
response sentence against the available document sentences. The score is forced to zero
when a numeric token in the response is absent from the source. The threshold is selected
on the validation split by maximizing coverage subject to an empirical false-certification
rate of at most 5%, then frozen and evaluated once on the test split. Labels and thresholds
from the test split are never used during selection.

This certificate concerns narrow source consistency only. It does not certify factual
truth, semantic entailment, clinical safety, or answer completeness.

---

## 11. Known Limitations

1. The oracle assumes that oracle-produced outputs at stage N are compatible inputs for stage N+1. If the oracle scope returns records that the pipeline's extraction model has never seen, the E-stage may produce systematically different outputs than expected. This is inherent to the 3-component counterfactual model.

2. Shapley computation requires all eight subset interventions to be valid. Missing or
   incompatible artifacts make the attribution unavailable and are counted as failures.

3. For text benchmarks, the `SupportStatus` loss mapping is a heuristic. It does not reflect any learned utility function. Two different queries with `SupportStatus.PARTIALLY_SUPPORTED` are treated as equal loss regardless of how much evidence was present.

4. The deterministic fixture extractor is not a trained model and produces rule-based results. Reported `support_status` from the fixture extractor reflects keyword overlap, not semantic understanding.

5. The SciFact deterministic baseline predicts one class for every example, so its
   classification score is a negative baseline rather than a competitive claim-verification
   result. The HotpotQA evidence-sentence baseline likewise has near-zero answer exact match.

6. The RAGBench certificate is validation-calibrated but not distribution-free. Its test
   false-certification rate can exceed the validation target, and the reported confidence
   intervals do not constitute a formal finite-sample risk guarantee.
