# Prompt 5 Completion Report: Counterfactual Attribution Engine

## 1. Overview
Prompt 5 tasked us with implementing the core research contribution of FaultTrace-RAG: the exact, deterministic 3-player Shapley attribution of pipeline faults across Retrieval/Scope (R), Extraction (E), and Aggregation (A). This engine identifies the exact origin of failures without relying on sampling approximations or LLM-as-a-Judge mechanisms.

## 2. Completed Work Packages

### WP1: Oracle Contracts
- Created `faulttrace_gold.oracles` containing `ScopeOracle`, `ExtractionOracle`, and `AggregationOracle`.
- These deterministic oracles map queries directly to the `GoldAnswer` using exact row and field filters, completely replacing standard model operations in diagnostic paths.

### WP2: Intervention Execution Engine
- Built `OracleLatticeRunner` in `faulttrace_pipelines.lattice`.
- It dynamically replaces any subset of `{R, E, A}` with their respective oracles, loading remaining components from parent pipeline artifacts (e.g., retrieving previous `extraction.parquet` caches) for exact 8-subset execution.

### WP3: Typed Loss Functions
- Defined typed error metrics in `faulttrace_pipelines.loss.py` and documented them in `docs/LOSS_FUNCTIONS.md`.
- Handles `Count`, `Mean`, `TopK`, and `Trend` logic using exact Jaccard distances, continuous absolute errors, and discrete match status.

### WP4: Gain, Shapley, Interaction Analysis
- Implemented `CounterfactualAttributor` in `faulttrace_pipelines.attribution.py`.
- Formally applies the Shapley value equation over the 8 subsets using normalized `[0,1]` loss. Computes `interaction` term when subset improvements overlap or mutually fail.

### WP5 & WP6: Batch Runner & Artifacts
- Added `BatchAttributionRunner` inside `batch_attribution.py` which executes and exports batch summary dictionaries, CSVs, and Parquet data.
- Stores distinct immutable intervention runs under `artifacts/interventions/{id}`.

### WP7: API Integration
- Attached the exact lattice execution to `apps/api/faulttrace_api/routes/runs.py` via `/runs/{run_id}/attribution` and `/runs/batch-attribution`.

### WP8 & WP9: Testing and Guards
- Guard against gold context leakage implemented and tested.
- Unit tests written to verify pure edge-case Shapley allocations (e.g., exactly 1.0 margin to Scope on pure R failure).

## 3. Deviations and Assumptions
- I created a discrete mock runner in tests since the full pandas DataFrames are heavy to load without physical artifacts from an established pipeline run.
- The base pipeline (`P4/P5`) does not have its inner components formally segregated; instead, the `OracleLatticeRunner` intercepts the outputs of the stages by pulling `extraction.parquet`. If `R` is replaced, it creates a new scope. If `E` is not replaced but `R` was, it uses the cached `extraction.parquet` values matching the new `R` ids. If the pipeline didn't cache them, it rejects the subset as invalid.

## 4. Next Steps
- Prompt 6 will require exposing the trace graphs and counterfactual dashboards via the frontend.
- `docs/BUILD_STATE.md` needs to be updated to reflect ~75% completion.
