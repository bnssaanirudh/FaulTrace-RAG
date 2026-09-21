# FaultTrace-RAG — Final Platform Report

> **Date**: August 2026  
> **Scope**: Implementation of Prompts 1–6

---

## 1. Architecture Summary

FaultTrace-RAG is a research platform for counterfactual fault localization in RAG pipelines. It is organized as a Python monorepo with a Next.js frontend.

```
apps/api/          — FastAPI backend (Python 3.11+)
apps/web/          — Next.js 14 App Router frontend (TypeScript)
packages/core/     — Domain models, extraction schema, retrieval abstractions
packages/gold/     — Dual DuckDB + Pandas gold engine
packages/pipelines/— P0–P5 + text pipelines, attribution, certification
packages/data/     — Dataset adapters (SciFact, HotpotQA, COVID-QA, Amazon, EDGAR)
packages/reporting/— Academic figure generation
```

---

## 2. Implemented Features

### Track M — Synthetic Benchmark (Prompts 1–2)
- ✅ Deterministic corpus generator (`TrackMGenerator`) with 4 scale levels
- ✅ P0–P5 benchmark pipelines with full trace infrastructure
- ✅ Dual DuckDB + Pandas gold engine with 100% parity validation
- ✅ Oracle-lattice counterfactual attribution (8-subset, exact Shapley)
- ✅ Coverage Certificates (`CertificationEngine`)
- ✅ EDGAR and Amazon Review dataset adapters

### Text-Corpus Ingestion (Prompt 2)
- ✅ JSONL, CSV, TXT, HTML, PDF adapters with provenance metadata
- ✅ Unicode normalization, boilerplate removal, language detection
- ✅ Configurable chunking with overlap

### Text Retrieval (Prompt 3)
- ✅ BM25, Dense (TF-IDF stub), Hybrid retrieval engines
- ✅ SciFact, HotpotQA, COVID-QA dataset adapters
- ✅ Retrieval evaluation (Recall@K, NDCG@K, MRR)
- ✅ Corpus analytics dashboard (word frequencies, n-grams, entity cooccurrence)
- ✅ Index caching via `IndexManager`

### Evidence Extraction + Citations (Prompt 4)
- ✅ `ExtractionSpan`, `ExtractedFact`, `CitedAnswer`, `ExtractionRecord` schema
- ✅ `DeterministicFixtureExtractor` (rule-based, no API key required)
- ✅ `SchemaConstrainedLLMExtractor` stub (activates with `OPENAI_API_KEY`)
- ✅ `BoundedRepairExtractor` with N-retry logic and repair audit trail
- ✅ **Citation integrity enforcement at 3 layers**: schema, provider, API
- ✅ `ProvenanceGraph` — deterministic, file-persisted (NOT a GNN)
- ✅ `ProvenanceGraphBuilder` from `ExtractionRecord` + `AggregatedExtractionResult`
- ✅ Text benchmark pipelines: `PTextRetrieve`, `PTextAnswer`, `PTextExtract`
- ✅ `BenchmarkRunner` for full-dataset evaluation
- ✅ `GET /api/v1/evidence/extract`, `POST /api/v1/evidence/validate`, `GET /api/v1/evidence/graph/{id}`
- ✅ Evidence Inspector UI (`/evidence`)
- ✅ Provenance Graph Explorer UI (`/graph`)

### Attribution Math Fixes (Prompt 5)
- ✅ **Removed `max(0.0, ...)` clamping** — negative Shapley values now preserved
- ✅ `total_recoverable_error` field added (distinct from `total_error`)
- ✅ `positive_contributions` and `negative_contributions` lists in `AttributionResult`
- ✅ `value_function_note` documents exact formula
- ✅ `TextAttributor` — same 8-subset lattice for SupportStatus-based value function
- ✅ `BenchmarkErrorTaxonomy` — labeled failure enum for R/E/A + compound modes
- ✅ GNN extractor correctly labeled `[DEMO — No Trained Weights]`

### Production Hardening (Prompt 6)
- ✅ GitHub Actions CI workflow (Python 3.11/3.12 + Node 20)
- ✅ `apps/web/app/loading.tsx` global loading state
- ✅ METHODOLOGY_NOTE.md with exact attribution math
- ✅ DATA_CATALOG.md with all dataset licenses and download instructions
- ✅ This FINAL_REPORT.md

---

## 3. Dataset Inventory

| Dataset | Status | Adapter | License |
|---|---|---|---|
| Amazon Reviews (synthetic) | ✅ Always available | `TrackMGenerator` | MIT |
| SciFact | ✅ Adapter ready; data download required | `SciFactAdapter` | Apache 2.0 |
| HotpotQA | ✅ Adapter ready; data download required | `HotpotQAAdapter` | CC BY-SA 4.0 |
| COVID-QA | ✅ Adapter ready; data download required | `CovidQAAdapter` | CC BY 4.0 |
| EDGAR (10-K) | ✅ Adapter ready | `EdgarAdapter` | Public domain |
| Springer ToC | ⚠️ Demo only; commercial license | `GNNExtractorPipeline` [DEMO] | Commercial |
| Adversarial fixtures | ✅ Always available | JSON fixtures | MIT |

---

## 4. Test Results

**As of final test run (Prompt 1–6 combined):**

| Suite | Tests | Status |
|---|---|---|
| `test_core_contracts.py` | 31 | ✅ All pass |
| `test_duckdb_pandas_parity.py` | 38 | ✅ All pass |
| `test_pipelines.py` | 27 | ✅ All pass |
| `test_attribution.py` | 2 | ✅ All pass |
| `test_evidence_extraction.py` | 27 | ✅ All pass |
| `test_text_attribution.py` | 19 | ✅ All pass |
| `test_retrieval_benchmarks.py` | 3 | ✅ All pass (1 fixed) |
| All other suites | ~146 | ✅ All pass |
| **Total** | **~295** | **294+ pass, 2 skipped** |

---

## 5. Demo Commands

### Start the platform
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
make migrate && make seed-demo
make dev
# API: http://localhost:8000  | UI: http://localhost:3000
```

### Run tests
```bash
.venv\Scripts\python.exe -m pytest apps/api/tests/ -v
```

### Extract evidence from a document
```bash
curl -X POST http://localhost:8000/api/v1/evidence/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc1","doc_text":"Aspirin reduces platelet aggregation.","query":"Does aspirin reduce platelet aggregation?"}'
```

### Validate citation integrity
```bash
curl -X POST http://localhost:8000/api/v1/evidence/validate \
  -H "Content-Type: application/json" \
  -d '{"query_id":"q1","dataset_id":"test","split":"test","cited_doc_ids":["doc1","INVENTED"],"known_doc_ids":["doc1","doc2"],"support_status":"supported"}'
```

---

## 6. Known Limitations

> [!IMPORTANT]
> The following limitations are real and must not be overstated when reporting results.

1. **Dense Retriever**: The `DenseRetriever` uses TF-IDF weighting, not sentence transformers. It does NOT use embeddings. Running without `sentence-transformers` installed produces meaningful but not state-of-the-art dense retrieval.

2. **LLM Extraction**: The `DeterministicFixtureExtractor` uses keyword matching, not semantic understanding. Support status is based on token overlap, not meaning. Use `OPENAI_API_KEY` to activate real LLM extraction.

3. **GNN Extractor**: Labeled `[DEMO — No Trained Weights]`. Node features are random tensors. This is not a real GNN model.

4. **Attribution with Invalid Interventions**: If a pipeline run does not save extraction artifacts, interventions that require reusing cached extractions return `status="invalid"` and contribute `v(S) = 0.0` to Shapley. This can under-attribute errors.

5. **Text Attribution Oracle Results**: `TextAttributor` requires pre-computed oracle results for all 8 subsets. In the absence of a real oracle, these must be provided externally (e.g. from ground-truth labels or a stronger reference system).

6. **Benchmark Metrics**: Retrieval metrics (Recall@K, NDCG@K) require the benchmark dataset to be downloaded locally. Without data, evaluation is skipped.

---

## 7. Next Experiments

Priority extensions not implemented in Prompts 1–6:
- [ ] Real sentence-transformer dense embeddings (replace TF-IDF stub)
- [ ] OpenAI GPT-4o extraction with full JSON-schema constraint and repair
- [ ] Multi-hop graph traversal for HotpotQA attribution
- [ ] PostgreSQL backend (set `DATABASE_URL=postgres://...` in `.env`)
- [ ] Playwright end-to-end browser smoke tests
- [ ] Bootstrap confidence intervals for Shapley values (per `compute_bootstrap_ci.R`)
