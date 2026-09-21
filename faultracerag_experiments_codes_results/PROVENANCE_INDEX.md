# Provenance Index

This document classifies the experimental artifacts within `faultracerag_experiments_codes_results` to ensure transparent provenance and prevent the accidental inclusion of exploratory or debugging outputs in the final manuscript.

> [!WARNING]
> **FAST_MODE results are exploratory/debugging outputs and must not be used for manuscript headline claims.**

## Artifact Classifications

| Path | Classification | Manuscript Eligible? | Notes |
|---|---|---|---|
| `04_final_results/` | `canonical` / `paper_mode` | Yes | Canonical results matching the final manuscript. |
| `03_raw_result_archives/fast_mode/` | `fast_mode` / `exploratory` | No | Exploratory and debugging outputs. Excluded from tables unless overridden. |
| `01_notebooks/FaulTrace_Phi4_Corrected_Rerun.ipynb` | `preliminary` / `excluded` | No | Phi-4 results were excluded due to generation-termination mismatch during audit. |

## Tooling Integration

The `scripts/analyze_verified_results.py` and downstream table-generation scripts have been updated to proactively reject artifacts tagged with `FAST_MODE` unless the `--allow-fast-mode` safety flag is provided. This acts as a mechanical guarantee that exploratory data cannot leak into the manuscript tables.

For machine-readable metadata, see `artifact_classification.json`.
