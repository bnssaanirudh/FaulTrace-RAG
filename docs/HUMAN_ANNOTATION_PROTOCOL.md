# Human Annotation Protocol: FaultTrace-RAG Natural-Failure Benchmark

This document specifies the protocol for independently annotating the FaultTrace-RAG natural-failure cases. The goal is to produce adjudicated ground-truth stage-level fault labels that can be used to evaluate the localization accuracy of the FaultTrace attribution engine.

## 1. Overview

**Target**: A stratified sample of 50–100 cases drawn from the 200 audited Qwen/Mistral live-failure cases.

**Task**: Assign one or more stage-level fault labels to each case, identifying which pipeline component caused the observed answer failure.

**Pipeline Stages (for annotation purposes)**:
| Stage | Label Code | Description |
|---|---|---|
| Retrieval/Scope | `R` | The wrong or incomplete set of records was retrieved. |
| Extraction/Evidence | `E` | Facts were incorrectly extracted from the retrieved records. |
| Aggregation/Reasoning | `A` | The model incorrectly aggregated or reasoned over correct facts. |
| Generation | `G` | The model produced an incorrect surface form despite correct reasoning. |
| Mixed/Compound | `COMPOUND` | Multiple stages jointly caused the failure; neither alone is sufficient. |
| Insufficient Information | `INSUF` | The case cannot be resolved with available evidence. |
| Annotation Uncertain | `UNK` | Annotator cannot determine the fault stage with confidence. |

## 2. Stratification

Sample the benchmark cases to ensure:
- **Model diversity**: At least 40% from each of Qwen-2.5 and Mistral-Large.
- **Dataset diversity**: SciFact, HotpotQA distractor, and RAGTruth each represented.
- **Severity balance**: Mix of minor (score > 0.5) and major (score ≤ 0.5) baseline errors.
- **Repair outcome balance**: Include both repaired and unrepaired cases.
- **Multihop**: At least 20% multihop queries.

## 3. Annotation Input Fields (per case)

Each case for annotation should include:
- `case_id`: Unique identifier linking back to the 200-case audit log.
- `model`: The LLM that produced the output (e.g., `Qwen-2.5-72B`).
- `query`: The analytical query.
- `retrieved_scope`: The records/passages retrieved.
- `extracted_facts`: The structured facts extracted from scope.
- `predicted_answer`: The model's final answer.
- `gold_answer`: The deterministic gold answer.
- `error_magnitude`: Numeric difference or EM mismatch.

## 4. Annotation Output Schema

Each annotation record should output:
```json
{
  "case_id": "...",
  "annotator_id": "A1",
  "primary_fault": "R",
  "secondary_faults": ["E"],
  "is_compound": false,
  "confidence": 3,
  "evidence_quote": "The retrieved records did not include record ID 4421 which contains the correct count.",
  "notes": "Retrieval clearly missed the relevant record."
}
```

**Confidence scale**:
- `3` = High (annotator is certain)
- `2` = Moderate (annotator is fairly confident)
- `1` = Low (annotator is guessing; consider labeling `UNK`)

## 5. Allowed Multilabel Combinations

| Primary | Allowed Secondary | Notes |
|---|---|---|
| `R` | `E`, `A` | Retrieval failure can cascade |
| `E` | `A`, `G` | Extraction error can corrupt downstream stages |
| `A` | `G` | Aggregation error can produce surface-level mistakes |
| `COMPOUND` | Any | Use when no single primary is sufficient |
| `INSUF` | — | Cannot be combined |
| `UNK` | — | Cannot be combined |

## 6. Edge Cases

- **Empty Scope**: If `retrieved_scope` is empty and the model answers incorrectly, annotate as `R`.
- **Correct Facts, Wrong Answer**: If `extracted_facts` are correct and the answer is wrong, annotate as `A` or `G`.
- **Partially Correct**: If some facts are correct and some are wrong, annotate `E` as primary with `A` as secondary.
- **Hallucinated Entity**: If the model introduces an entity not in the scope, annotate as `A`.
- **Correct Answer, Different Form**: If the answer is semantically correct but the wrong surface form (e.g., "3 million" vs "3,000,000"), annotate as `G`.

## 7. Inter-Annotator Agreement

- **Minimum annotators**: 2 independent annotators per case.
- **Agreement metrics**:
  - Raw agreement rate (percentage of cases where labels match exactly)
  - Cohen's kappa for single-label primary fault decisions
  - Jaccard similarity for multilabel secondary fault sets
- **Adjudication**: Cases with disagreement are adjudicated by a third reviewer.

## 8. Adjudication Policy

- If two annotators agree on the primary label, it is adopted without adjudication.
- If annotators disagree on primary label, a third annotator reviews the case and their judgment is final.
- If a third reviewer also cannot resolve the disagreement, the case is labeled `COMPOUND` or `UNK` based on the nature of the disagreement.

## 9. What Constitutes a Human Annotation

> [!CAUTION]
> LLM-generated fault labels are **NOT** human annotations, even if prompted carefully. Human annotations require a human annotator to read the evidence and make an independent judgment. LLM outputs may be used as a secondary reference during the adjudication phase but must never be counted as the primary annotation source.

## 10. Evaluating Localization Accuracy

Once adjudicated labels are available, evaluate FaultTrace-RAG as follows:

1. **Exact Set Accuracy**: Percentage of cases where FaultTrace's predicted fault set exactly matches the adjudicated label set.
2. **Macro F1**: Per-stage F1 averaged equally across all stage labels.
3. **Micro F1**: Stage-level precision/recall across all cases.
4. **Intervention Response**: Whether applying the attributed repair flips the answer.
5. **Annotator Agreement**: Report kappa and Jaccard before and after adjudication separately.

Report localization accuracy, intervention response, and annotator agreement as three distinct metrics — do not conflate them.

## 11. Output Files

Annotation results should be saved in:
- `research/canonical/annotations/raw_annotations.jsonl` — One record per annotation.
- `research/canonical/annotations/adjudicated.jsonl` — One adjudicated record per case.
- `research/canonical/annotations/agreement_stats.json` — Computed kappa, Jaccard, and raw agreement.
