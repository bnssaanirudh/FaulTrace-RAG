# Counterfactual Attribution Interpretation

This document provides researchers with boundaries and assumptions for interpreting the output of the Counterfactual Attribution Engine (Prompt 5).

## The Decomposition Model
We decompose RAG fault attribution into three components using exact Shapley values derived from oracle replacement:
- **R (Scope)**: The subset of corpus records supplied to the extraction engine.
- **E (Extraction)**: The structured facts drawn from the supplied scope.
- **A (Aggregation)**: The reduction operation applied to the extracted facts.

## Exact Shapley Efficiency
The Shapley values (`phi_R`, `phi_E`, `phi_A`) sum precisely to the "recoverable loss" (the difference between the baseline pipeline's error and the full-oracle error). Any residual loss that cannot be addressed by an individual component is assigned to the **interaction term**.

## Assumptions and Limitations
1. **No External Causality**: These values explain *how* the pipeline failed within the constraints of the 3-component model. They do not diagnose *why* a particular component failed (e.g., bad model weights, bad chunking).
2. **Oracle Validity**: We assume the `ScopeOracle`, `ExtractionOracle`, and `AggregationOracle` represent the absolute truth.
3. **Monotonicity Not Assumed**: A repair in one component (e.g. Scope) might actually *increase* the error in the final answer if the Extraction component is tuned to compensate for Scope errors. Our Exact Shapley evaluation permits negative marginal contributions.

## Interactions
If `interaction > 0`, it indicates a complex failure where two or more components mutually rely on each other's errors, and fixing one individually yields limited or zero improvement.

## Leak Guards
Gold information is strictly quarantined to `faulttrace_gold.oracles` and must not enter the `ProviderConfig` prompt generation logic under any circumstances. Automated leak checks in the test suite verify that `record_ids` and `GoldAnswer` objects remain absent from standard pipeline runs.
