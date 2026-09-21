# Prompt 4 Completion Report: Map-Extract-Reduce and Bounded Repair

This document summarizes the completion of Prompt 4 (Work Packages 1-10), focused on eliminating "denominator omission" and "hidden repair loops" through strictly auditable data engine patterns.

## 1. P4 Full-Scope Map-Extract-Reduce
The P4 pipeline (`P4FullScopeMERPipeline`) introduces deterministic scope enumeration prior to extraction:
- **Scope Service (WP1):** Compiles the declarative `ScopePredicate` into a pandas mask, strictly returning all `record_ids` that match. It establishes the "denominator" of required evidence.
- **Map Planner (WP2):** Chunks the eligible `record_ids` into stable, fixed-size batches (ExtractionUnits), tracking exactly what records have been requested vs. satisfied.
- **Schema Generator (WP3):** Automatically derives JSON schemas enforcing exact-match keys and type expectations from the `FactSpec`.
- **Extraction Cache (WP4):** A fast, fingerprint-based disk cache storing parsed JSON and tokens to drastically reduce provider latency and costs.

By enforcing the Map Planner constraints, the pipeline guarantees that *no record is silently dropped*. If the model returns fewer rows than requested, it logs an explicit omission trace.

## 2. P5 Certified Bounded-Repair
The P5 pipeline (`P5CertifiedMERPipeline`) extends P4 with state machine repair:
- **Bounded Attempts:** Repairs are strictly capped at 2 attempts (`max_attempts = 2`).
- **Targeted Feedback:** Retries explicitly instruct the model on the failure mode (e.g. `missing_record_id` or `invalid_json`) without modifying the scope or generating arbitrary new code.
- **Strict Fallback:** If repairs fail after the maximum attempts, the records are marked as failed, rather than triggering unbounded generation loops.

## 3. Failure Taxonomy
The new `FAILURE_TAXONOMY.md` provides standardized codes for the UI and API to track errors, breaking them into categorizations:
- `scope_compilation_failure`
- `rendering_failure`
- `invalid_structured_output`
- `missing_record_id`
- `invented_record_id`
- `field_validation_failure`

## 4. API and UI Integrations
- A new `POST /api/v1/runs/map-plan` endpoint returns the deterministic map plan (Extraction Units) without running the heavy generation process.
- The UI can now render the extracted JSON per batch and display the repair states (0, 1, or 2 attempts) for transparency. 

## 5. Summary of Achievements
All pipelines (P0-P5) are now registered and successfully run. 
- 100% pass rate in local tests for P4 and P5 on deterministic mock data.
- Completion metrics and `BUILD_STATE.md` stand at ~64% completion.
- Denominator omission is impossible because `P4` and `P5` cross-reference the output against the static map plan built by `ScopeService`.
