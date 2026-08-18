# Benchmark Integration Status

This document describes the current state of text-retrieval benchmark integrations in FaultTrace-RAG.

## Fully Supported Benchmarks

### SciFact (BEIR Format)
- **Status:** Supported
- **Adapter:** `packages/data/faulttrace_data/benchmarks/scifact.py`
- **Description:** Scientific fact-checking corpus containing claims and corresponding abstracts.
- **Support Details:** Corpus ingestion, query parsing, and extraction validation are fully integrated.

## Planned / Pending Benchmarks

### HotpotQA
- **Status:** Planned
- **Description:** Multi-hop question answering dataset.
- **Notes:** Currently present in the UI as a distractor. Integration is planned for a future release to validate multi-document reasoning traces.

### COVID-QA (RAGBench)
- **Status:** Pending
- **Description:** Question answering about COVID-19 related scientific articles.
- **Notes:** Partial integration started. Full integration pending pipeline evaluation.
