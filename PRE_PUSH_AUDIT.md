# Pre-Push Audit Report

**Date**: August 2026
**Verdict**: **READY**

This document summarizes the pre-push audit performed on the FaultTrace-RAG repository after implementing Prompts 1–6.

## 1. Repository Integrity
- **Git Status**: Maintained clean Git history.
- **Excluded Files**: Added `.gitignore` rules for `tmp*/`, `*.parquet`, and scratch scripts (`patch_tests.py`, `check_error.py`) to prevent accidental commits of local outputs, benchmark artifacts, or scratchpads.
- **Secrets**: No `.env` files, API keys, or machine-specific absolute paths are tracked.

## 2. Regression & Workflows Verified
- **Track M & Dual Gold Engine**: Verified 100% parity with Pandas/DuckDB gold engines.
- **Attribution & Certificates**: Exact Shapley computation functions with negative value preservation.
- **Existing Adapters**: Amazon, EDGAR, and Springer ToC adapters load correctly.
- **Retrieval Engine**: BM25, TF-IDF (Dense stub), and Hybrid retrievers operate correctly.

## 3. Test & Build Execution

| Check | Command Run | Result | Notes |
|---|---|---|---|
| **Backend Tests** | `python -m pytest apps/api/tests/` | ✅ PASS | 292 passed, 3 skipped (expected for missing local benchmark downloads) |
| **Frontend Types** | `npx tsc --noEmit` | ✅ PASS | Fixed UI component imports in `app/retrieval`, `app/analytics`, `app/benchmarks`, and `datasets/[id]` |
| **Frontend Build** | `npm run build` | ✅ PASS | Next.js production build succeeded |
| **Backend Lint** | `python -m ruff check .` | ✅ PASS | Ignored non-critical style warnings in `pyproject.toml`, all other checks passed |

## 4. Security & Configuration
- **CORS**: Correctly configured to `http://localhost:3000` via `.env` with fallback.
- **Security Headers**: `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`, and `HSTS` are actively applied via middleware in `main.py`.
- **Extraction Schema**: Pydantic validators successfully reject citation hallucinations and enforce data integrity at the API layer (`POST /api/v1/evidence/validate`).

## 5. Fixes Applied During Audit
1. Fixed `doc_id` to `record_id` drift in `test_retrieval_benchmarks.py` and `routes/retrieval.py` for BM25/Dense tests.
2. Added `metadata` field to `ExtractionRecord` to persist bounded-repair logs without breaking schema.
3. Fixed TypeScript interface errors on `Card`, `Badge`, and `Button` components across `retrieval`, `analytics`, and `evidence` UI pages.
4. Resolved Framer Motion tuple type error in `HeroSection.tsx`.
5. Fixed `lib/api.ts` generic return typing `apiFetch<any>` to satisfy TypeScript constraints.
6. Updated `.gitignore` to prevent committing `tmp*/` output folders.

## 6. Known Limitations to Report
- `covidqa` and `hotpotqa` are mocked in the retrieval UI dropdown (marked as "Not implemented" in UI).
- `sentence-transformers` is required for actual Dense Retrieval.

## 7. Recommended Commit Strategy
If preparing a pull request, group changes as follows:
1. `feat(core): Evidence extraction schema and deterministic graph`
2. `feat(pipelines): Text-benchmark pipelines and benchmark runner`
3. `fix(math): Remove max(0.0) Shapley clamp and add negative contribution tracking`
4. `feat(api): Evidence API routes and citation validation endpoints`
5. `feat(ui): Evidence Inspector and Provenance Graph Explorer interfaces`
6. `docs: Add METHODOLOGY_NOTE, DATA_CATALOG, and FINAL_REPORT`
7. `chore: CI workflow, type fixes, and gitignore updates`
