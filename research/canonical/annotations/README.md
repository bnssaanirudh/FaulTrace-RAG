# Annotation Status

This directory will contain the human-annotated stage-level fault labels for the FaultTrace-RAG natural-failure benchmark.

## Status: PENDING

Human annotation has not yet been executed. The protocol is documented in `docs/HUMAN_ANNOTATION_PROTOCOL.md`.

## Steps to Complete

1. Sample 50–100 cases from `05_live_llm_evidence/validated_evidence/` using the stratification criteria in the protocol.
2. Distribute to two independent annotators using the annotation schema in the protocol.
3. Compute inter-annotator agreement (Cohen's κ, Jaccard).
4. Adjudicate disagreements per the adjudication policy.
5. Save adjudicated results to `adjudicated.jsonl` (replacing the empty scaffold).
6. Save raw per-annotator records to `raw_annotations.jsonl`.
7. Save agreement statistics to `agreement_stats.json`.
8. Run `python scripts/run_baseline_comparison.py --cases research/canonical/annotations/adjudicated.jsonl --baseline ragchecker` to produce the comparison table.

## File Schema

Each line in `adjudicated.jsonl` must match the schema:
```json
{
  "case_id": "case-001",
  "model": "Qwen-2.5-72B",
  "dataset": "HotpotQA",
  "query": "...",
  "retrieved_scope": [...],
  "extracted_facts": [...],
  "predicted_answer": "...",
  "gold_answer": "...",
  "error_magnitude": 0.85,
  "adjudicated_faults": ["R"],
  "annotator_agreement": "agreed",
  "kappa": 0.82
}
```
