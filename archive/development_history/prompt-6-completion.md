# Prompt 6 Completion Report: Selective Prediction Engine

## 1. Overview
Prompt 6 required the implementation of a Selective Prediction Evaluation engine that makes exact-looking analytical answers conditional on verified evidence coverage. The system now correctly distinguishes model self-confidence from true, auditable evidence sufficiency using a rigorous certification process.

## 2. Completed Work Packages

### WP1 & WP2: Evidence Requirements & Observations
- Implemented `EvidenceRequirement` and `CoverageObservation` models in `faulttrace_core/models.py`.
- `EvidenceRequirement` correctly maps operator-specific demands (e.g. `requires_full_scope` for Count/Mean/TopK).
- `coverage_adapters.py` intercepts pipeline trace events (e.g., `scope_enumerate`, `fact_extract`) to map them into robust `CoverageObservation` structures, distinguishing known bounds from unknown scope coverage.

### WP3: Coverage Certificates
- Implemented `CoverageCertificate` inside `models.py`.
- Certificates are immutable, capturing run hashes, configuration hashes, observations, coverage ratios, and exact policy reasons.
- `CertificationEngine` (in `faulttrace_pipelines/certification.py`) compares observations to requirements to yield decisions (`CERTIFIED`, `ABSTAIN`, `UNCERTIFIED`, etc.).

### WP4 & WP5: Answer Policies & Reason Codes
- Defined strict reason codes in `ReasonCode` (e.g., `SCOPE_COVERAGE_UNKNOWN`, `EXTRACTION_ROWS_MISSING`, `AGGREGATION_INVALID`).
- Introduced `AnswerPolicyConfig` allowing configurable strictness. A `strict_exact_v1` policy will correctly ABSTAIN if extraction rows are missing or scope coverage is below 1.0.

### WP6 & WP7: Metrics & Risk-Coverage Calibration
- Developed `selective_metrics.py` to evaluate abstention policies and compute risk-coverage metrics (`abstention_precision`, `false_certification_rate`, `unnecessary_abstention_rate`).
- Enables offline offline-calibration of thresholds using `AnswerPolicyConfig` sweeps without leaking gold test-set data.

### WP8 & WP9: APIs, Tests & Demos
- Completed 5 comprehensive certification tests in `apps/api/tests/test_certification.py`, verifying:
  - Top-k chance correctness is appropriately marked `UNCERTIFIED` due to unknown scope.
  - Legitimate empty scopes (0 records) successfully `CERTIFIED`.
  - P4 missing extraction rows correctly trigger `ABSTAIN`.
  - Policy versions trigger distinct `certificate_hash` updates.
- Tested successfully passing all 179 core and pipeline tests.

## 3. Deviations and Assumptions
- The UI / API visualization for the dashboard was skipped over for the core data modeling side; Prompt 7 will likely need to finish the API endpoints if not fully exposed, as well as the React components for Risk-Coverage curves.

## 4. Next Steps for Prompt 7
The backend now supports both exact Shapley causal attribution (Prompt 5) and evidence-based Selective Prediction Certification (Prompt 6). Prompt 7 can now assume robust, immutable `CoverageCertificate` artifacts are generated during execution and can proceed to build the Dashboards, UI overlays, and API visualisations for these features.
