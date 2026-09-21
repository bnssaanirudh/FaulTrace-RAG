# Baseline Comparison: FaultTrace-RAG vs. Contemporary Diagnostic Methods

This document describes the methodology, implementation notes, and results for the comparative evaluation of FaultTrace-RAG against strong contemporary RAG diagnostic baselines.

## 1. Baseline Selection

After reviewing the novelty matrix (`paper/novelty_matrix.md`), we select the following baselines for comparison:

| Baseline | Rationale | Source |
|---|---|---|
| **RAGChecker** | Component-level precision/recall diagnostics; most direct structural comparator | Ru et al., 2024 |
| **RAGAS (Faithfulness + Answer Relevancy)** | Widely-used LLM-judge-based pipeline scoring | Es et al., 2023 |
| **LLM Judge (GPT-4o chain-of-thought)** | Upper-bound approximation for LLM-based component labeling | — |

> [!IMPORTANT]
> We do **not** choose weak baselines. These represent the current state-of-the-art for RAG component diagnosis.

## 2. Matching Requirements

All baselines are evaluated under identical conditions:
- **Dataset**: Same 200 Qwen/Mistral natural-failure cases used in Experiment EXP-002.
- **Queries**: Identical queries, retrieved scopes, and extracted facts.
- **Gold Labels**: Adjudicated human stage-level annotations (see `docs/HUMAN_ANNOTATION_PROTOCOL.md`).
- **Token Budget**: Baselines that use LLM judges are given the same token allowance per case.
- **No Test-Time Tuning**: Baselines are applied zero-shot with their published configurations.

## 3. Metrics Reported

For each baseline and FaultTrace-RAG, report:

| Metric | Description |
|---|---|
| Exact Set Accuracy | % of cases with exactly matching fault set |
| Macro F1 | Per-stage F1 averaged equally |
| Micro F1 | Stage-level P/R/F1 across all cases |
| Per-Stage P/R/F1 | Breakdown by R, E, A, G stages |
| Compound-Fault Accuracy | Accuracy on multi-stage cases only |
| Latency (s/case) | Mean wall-clock time per case |
| Token Usage | Mean input+output tokens per case |
| Monetary Cost ($/case) | Estimated API cost where applicable |

For Active Diagnosis (BACD):
- **Budget-Matched Comparison**: Compare FaultTrace BACD vs. baseline at equal probe counts (3, 5, 7 probes).

## 4. Key Results Summary (Template)

> **Note**: This template is filled from the comparison experiment run. Actual numerical values must be populated from the experiment artifacts — do not fill from this document.

| Method | Exact Set Acc | Macro F1 | Micro F1 | Latency | Tokens/case |
|---|---|---|---|---|---|
| FaultTrace-RAG (Shapley) | — | — | — | — | — |
| RAGChecker | — | — | — | — | — |
| RAGAS | — | — | — | — | — |
| LLM Judge (GPT-4o) | — | — | — | — | — |

## 5. Oracle Access Disclaimer

FaultTrace-RAG uses stage oracles (ground-truth scope, extraction, and aggregation outputs) to compute Shapley values. RAGChecker and RAGAS do not have access to equivalent oracle information. This oracle advantage must be stated explicitly in the paper and is not a fair comparison for zero-oracle methods.

The BACD active-diagnosis comparison is oracle-free and represents the fairest comparison: FaultTrace adaptively selects probes without accessing gold labels, and this is compared against baselines at the same probe/cost budget.

## 6. Statistical Analysis

- Use **paired Wilcoxon signed-rank tests** for per-case matched comparisons.
- Report **bootstrap 95% confidence intervals** (10,000 resamples) for all aggregate metrics.
- Differences are reported as significant only if p < 0.05 and the CI does not include zero.

See `scripts/compute_bootstrap_ci.R` for the implemented bootstrap analysis.

## 7. Failure Cases

An honest analysis must document cases where FaultTrace-RAG performs worse than baselines. Document:
- Cases where compound faults were incorrectly attributed to a single stage.
- Cases where oracle access would be unfair to compare.
- Cases where low token budgets caused FaultTrace to underperform.

## 8. Methodology Documentation

The baseline adapter implementations are located in:
- `packages/pipelines/faulttrace_pipelines/baselines/ragchecker_adapter.py` (planned)
- `packages/pipelines/faulttrace_pipelines/baselines/ragas_adapter.py` (planned)

Experiment configuration for the comparison sweep:
- `configs/baseline_comparison.yaml` (planned)
