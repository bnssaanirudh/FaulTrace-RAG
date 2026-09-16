# Benchmark Integration Status

This document describes the current state of text-retrieval benchmark integrations in FaultTrace-RAG.

## Measured retrieval integrations

### SciFact (BEIR Format)
- **Status:** Full-corpus retrieval measured on the complete local BEIR snapshot
- **Adapter:** `packages/data/faulttrace_data/benchmarks/scifact.py`
- **Description:** Scientific fact-checking corpus containing claims and corresponding abstracts.
- **Support Details:** Corpus ingestion, query/qrel parsing, BM25, dense, and hybrid retrieval.
- **Additional audit:** The official dev split has a deterministic claim-status baseline
  and 960 injected R/E/A attribution lattices. The classifier predicts `supported` for
  every claim; this is a mechanics baseline, not a competitive claim-verification model.

### HotpotQA
- **Status:** Complete distractor validation split measured as supplied-candidate retrieval
- **Adapter:** `packages/data/faulttrace_data/benchmarks/hotpotqa.py`
- **Description:** Multi-hop question answering dataset.
- **Notes:** Each question is ranked only over its supplied distractor contexts. A separate
  deterministic evidence-sentence baseline exercises downstream extraction and structural
  certification, but its answer EM is 0.00027. This is not fullwiki/open-domain retrieval
  and does not validate learned multi-hop answer generation.

### COVID-QA (RAGBench)
- **Status:** Complete test split measured as supplied-candidate retrieval
- **Adapter:** `packages/data/faulttrace_data/benchmarks/covidqa.py`
- **Description:** Question answering about COVID-19 related scientific articles.
- **Notes:** Each row is ranked over its supplied documents. Four rows without positive
  relevance labels are unscored for retrieval. A separate lexical/numeric certificate is
  calibrated on validation and evaluated on all 246 test responses. It is not a semantic
  entailment or truth certificate.

## Canonical result

The measured artifact is `artifacts/verified/20260828_external_retrieval_v4_final`.
Its `summary.json`, per-query Parquet file, dataset manifests, and checksum manifest
separate complete external data from the small hand-authored fixtures under
`apps/api/tests/fixtures/`.

The separate deterministic mechanics artifact is
`artifacts/verified/20260828_external_e2e_v4_final`. It contains 8,911 per-case rows,
source manifests, and a checksum manifest. Its global claim boundary is external pipeline
mechanics and narrow source consistency only; it is not a production-LLM or general
semantic-correctness evaluation.

The browser UI still exposes only SciFact retrieval. Adapter/evaluation support must not
be confused with a completed frontend or end-to-end product path for HotpotQA/COVID-QA.
