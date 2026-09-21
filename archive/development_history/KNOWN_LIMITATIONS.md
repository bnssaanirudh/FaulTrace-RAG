# Known Limitations: FaultTrace-RAG Prompt 1

## KL-001: Synthetic Track M Corpus (Not Real Amazon Data)

**Severity**: Research scope, not a defect  
**Description**: The Track M corpus used in Prompt 1 is a deterministic synthetic generator that mirrors the structure of Amazon review metadata (categories, ratings, brands, timestamps, prices, review text). It does NOT download or process the actual Amazon Reviews dataset.  
**Reason**: The full Amazon dataset is large, requires licensing agreement, and downloading it would block offline/demo use. Synthetic data is sufficient for testing gold engine correctness, pipeline traceability, and UI functionality.  
**Resolution path**: Prompt N will add an adapter that reads real Amazon metadata JSONL files when they are placed in `data/raw/`. The generator schema is designed to be compatible.

---

## KL-002: No Real LLM by Default

**Severity**: Expected; research scope  
**Description**: Pipeline P0 (deterministic scope baseline) uses no language model. It directly applies the scope predicate, extracts fact rows, and runs deterministic aggregation. Pipelines P1-P5 (direct-context, BM25, dense retrieval, oracle scope, full oracle) are not implemented in Prompt 1.  
**Reason**: Prompt 1 establishes the scaffolding, data, gold engine, and tracing infrastructure. Real LLM integration is Prompt 2-3 scope.  
**Resolution path**: Add `ModelProvider` implementations for Ollama, OpenAI-compatible endpoints, and local HuggingFace models in Prompt 2.

---

## KL-003: No Full Oracle Replacement Lattice

**Severity**: Research scope  
**Description**: The CoverageCertificate model is implemented as a Pydantic schema but the full 7-element oracle replacement lattice (R, E, A, RxE, RxA, ExA, RxExA) is not computed in Prompt 1. Only basic exact/tolerance correctness is computed.  
**Reason**: Oracle replacement requires multiple pipeline variants and LLM calls, which are Prompt 3 scope.  
**Resolution path**: Add `OracleReplacer` components in Prompt 3 that substitute each component and re-run the pipeline.

---

## KL-004: No Paper-Level Experiment Results

**Severity**: Expected  
**Description**: No experiment sweep results, no paper tables, no statistical significance tests are produced in Prompt 1.  
**Reason**: Experiments require the full pipeline suite and are planned for Prompt 4-5.  
**Resolution path**: Add `faulttrace_reporting` sweep runner in Prompt 4.

---

## KL-005: Frontend Requires Node.js 20+

**Severity**: Setup dependency  
**Description**: The Next.js frontend requires Node.js 20 or later. If Node.js is not installed, the frontend cannot be started.  
**Resolution path**: Install Node.js via `winget install OpenJS.NodeJS.LTS` or download from https://nodejs.org. The backend API works independently without Node.js.

---

## KL-006: Python 3.14 Compatibility

**Severity**: Low  
**Description**: The system targets Python 3.11+. The development environment uses Python 3.14 which is a pre-release-era version. Some packages may have compatibility warnings.  
**Resolution path**: Pin package versions that are known compatible. All core functionality (FastAPI, Pydantic v2, SQLAlchemy 2, DuckDB) works on Python 3.11-3.14.

---

## KL-007: Single-User SQLite Concurrency

**Severity**: Research limitation  
**Description**: SQLite supports only one writer at a time. Under concurrent pipeline runs, writes are serialized. This is acceptable for lab-scale research use.  
**Resolution path**: Set `DATABASE_URL` to a PostgreSQL connection string in `.env` to upgrade to full multi-writer support.

---

## KL-008: No Authentication or Authorization

**Severity**: Expected for research tool  
**Description**: The FastAPI service and Next.js dashboard have no authentication. All endpoints are publicly accessible within the configured CORS origin.  
**Resolution path**: Not planned; this is a research tool. If deployment is needed, add API key middleware or OAuth2.
