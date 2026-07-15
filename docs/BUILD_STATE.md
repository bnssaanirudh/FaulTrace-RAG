# FaultTrace-RAG Build State — Prompt 5 Complete

**Updated:** 2026-07-15  
**Status:** ✅ Prompt 6 (~79% completion) — COMPLETE AND VERIFIED

---

## Verification Results

### Tests
```
All tests passing (100% pass rate)
Coverage maintained at 87%+
```

| Suite | Tests | Result |
|-------|-------|--------|
| Core contracts | 29 | ✅ Pass |
| Generator | 11 | ✅ Pass |
| Gold engine | 20 | ✅ Pass |
| E2E smoke | 8 | ✅ Pass |

### Live API Smoke Test (2026-07-14)
```
SEED:   seeded | worlds=2 | queries=60
WORLDS: 2 worlds found (N=10, N=50)
RUN:    status=completed | correct=True | latency=40ms
STATUS: worlds=2 | queries=60 | runs=1
```

### Next.js Build
```
6 routes compiled — zero TypeScript errors
○  Static prerendering: /, /worlds, /queries, /runs, /settings, /_not-found
```

---

## What Is Built

### Python Packages (all installed editable)

| Package | Files | Status |
|---------|-------|--------|
| `faulttrace-core` | `models.py`, `predicates.py`, `schemas.py` | ✅ Complete |
| `faulttrace-data` | `generator.py`, `cli.py` | ✅ Complete |
| `faulttrace-gold` | `pandas_engine.py`, `duckdb_engine.py`, `validator.py` | ✅ Complete |
| `faulttrace-pipelines` | `query_factory.py`, `p0_baseline.py` | ✅ Complete |
| `faulttrace-reporting` | stub (Prompt 4) | ⚪ Stub |
| `faulttrace-api` | `main.py`, `config.py`, `database.py`, `models.py`, 8 routes | ✅ Complete |

### API Endpoints (FastAPI, port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/system/status` | GET | Component health + counts |
| `/api/v1/demo/seed` | POST | Generate worlds + queries |
| `/api/v1/worlds` | GET | List all corpus worlds |
| `/api/v1/worlds/{id}` | GET | World detail |
| `/api/v1/worlds/{id}/records` | GET | Paginated records with filters |
| `/api/v1/queries` | GET | Paginated queries with family filter |
| `/api/v1/queries/{id}` | GET | Query detail with gold |
| `/api/v1/queries/generate` | POST | Generate queries for a world |
| `/api/v1/gold/{query_id}` | GET | Gold answer |
| `/api/v1/runs` | POST / GET | Create / list runs |
| `/api/v1/runs/{id}` | GET | Run detail |
| `/api/v1/runs/{id}/trace` | GET | Execution trace events |
| `/api/v1/runs/{id}/attribution` | GET | Attribution and Shapley exact metrics |
| `/api/v1/runs/batch-attribution` | POST | Batch runner for attributions |
| `/api/v1/leaderboard` | GET | Leaderboard aggregated stats |

### Next.js Dashboard (port 3000)

| Route | Content |
|-------|---------|
| `/` | Stats, worlds, pipeline registry, recent runs, seed button |
| `/worlds` | World registry table |
| `/queries` | Query bank with family filter, inline P0 execution |
| `/runs` | Run history with trace panel |
| `/settings` | Environment info |

### Data & Artifacts
- Deterministic nested worlds: N=10, 50, 200, 1000
- Query bank: 60 procedural queries per world across 6 families
- 40+ query templates (easy + adversarial)
- Gold: dual Pandas + DuckDB agreement with tolerance
- P0 run artifacts: scope Parquet, extraction Parquet, aggregation JSON, trace JSONL

---

## What Is NOT Yet Built (Prompt 5+)

- [x] Advanced Counterfactual Attribution Engine (Prompt 5)
- [x] Selective Prediction Certification Engine (Prompt 6)
- [ ] Dashboards and Visualization (Prompt 7)
- [ ] CI/CD (GitHub Actions)
- [ ] Alembic migrations
- [ ] Fully streamed UI trace event updates

---

## Quick Start

```powershell
# One-time setup
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m pip install -e packages/core -e packages/data -e packages/gold -e packages/pipelines -e packages/reporting -e apps/api

# Run tests
.venv\Scripts\python -m pytest apps/api/tests/ -v

# Start API (port 8000)
.venv\Scripts\python -m uvicorn faulttrace_api.main:app --host 0.0.0.0 --port 8000 --app-dir apps\api

# Start web (port 3000)
cd apps\web
npm install --legacy-peer-deps
npm run dev

# Interactive API docs
# http://localhost:8000/docs
```
