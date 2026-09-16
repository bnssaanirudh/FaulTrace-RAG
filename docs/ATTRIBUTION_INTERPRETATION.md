# Counterfactual Attribution Interpretation

This document provides researchers with boundaries and assumptions for interpreting the output of the Counterfactual Attribution Engine (Prompt 5).

## The Decomposition Model
We decompose RAG fault attribution into three components using exact Shapley values derived from oracle replacement:
- **R (Scope)**: The subset of corpus records supplied to the extraction engine.
- **E (Extraction)**: The structured facts drawn from the supplied scope.
- **A (Aggregation)**: The reduction operation applied to the extracted facts.

## Exact Shapley Efficiency
The Shapley values (`phi_R`, `phi_E`, `phi_A`) sum to the recoverable loss (the
difference between baseline error and full-oracle error). The legacy `interaction` field is
only the floating-point efficiency residual and should be zero for a valid complete lattice.

## Assumptions and Limitations
1. **No External Causality**: These values explain *how* the pipeline failed within the constraints of the 3-component model. They do not diagnose *why* a particular component failed (e.g., bad model weights, bad chunking).
2. **Oracle Validity**: We assume the `ScopeOracle`, `ExtractionOracle`, and `AggregationOracle` represent the absolute truth.
3. **Monotonicity Not Assumed**: A repair in one component (e.g. Scope) might actually *increase* the error in the final answer if the Extraction component is tuned to compensate for Scope errors. Our Exact Shapley evaluation permits negative marginal contributions.

## Interactions
This implementation does not estimate a separate Shapley interaction index. Component
dependence can be inspected in the eight intervention losses, but the efficiency residual
must not be presented as a causal interaction effect.

## Three Diagnostic Views

- `dominant_fault` is the single component with the largest absolute Shapley value.
- `outcome_active_faults` contains components whose absolute Shapley contribution exceeds
  the declared numerical or calibrated threshold. It concerns answer-loss contribution.
- `artifact_discrepancy_faults` compares persisted R/E/A artifacts against stage-matched
  oracle outputs. E* receives the observed scope and A* receives the observed facts, so an
  upstream discrepancy is not automatically double-counted downstream.

The artifact view is offline and oracle-assisted. It can expose a component mismatch hidden
by downstream masking, but it is neither an online certificate nor unrestricted causal
identification. A declared injection may also be observationally silent; reports must
distinguish attempted injection, realized artifact discrepancy, and answer-loss effect.

## Leak Guards
Gold information is strictly quarantined to `faulttrace_gold.oracles` and must not enter the `ProviderConfig` prompt generation logic under any circumstances. Automated leak checks in the test suite verify that `record_ids` and `GoldAnswer` objects remain absent from standard pipeline runs.
