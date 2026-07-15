# FaultTrace-RAG

## Counterfactual Fault Localization for Corpus-Level LLM Analytics Pipelines

FaultTrace-RAG addresses a fundamental challenge in LLM-based analytics: when a pipeline produces a wrong answer over a corpus, **which component failed?** Was it the retrieval scope (wrong evidence set R), fact extraction (incorrect E), aggregation logic (wrong A), or an interaction among these?

### Research Problem

Given a corpus-level query Q and a pipeline producing answer A* != A_gold, FaultTrace-RAG computes:

- **Scope Error (R)**: Did the pipeline retrieve the wrong records?
- **Extraction Error (E)**: Did it extract incorrect structured facts?
- **Aggregation Error (A)**: Did it compute the aggregate incorrectly?
- **Interaction Effects**: R x E, R x A, E x A, R x E x A combinations

By systematically replacing each component with a deterministic oracle, the system attributes recoverable errors and issues coverage certificates.

---

## System Components (Current: Prompt 1 / ~30%)

| Component | Status |
|-----------|--------|
| Core domain contracts | IMPLEMENTED |
| Track M corpus generator | IMPLEMENTED |
| Procedural query factory | IMPLEMENTED |
| Dual gold engine (Pandas + DuckDB) | IMPLEMENTED |
| Deterministic baseline pipeline (P0) | IMPLEMENTED |
| FastAPI service | IMPLEMENTED |
| Next.js dashboard | IMPLEMENTED |
| Real LLM providers (P1-P5) | NOT YET |
| Oracle replacement lattice | NOT YET |
| Counterfactual attribution scores | NOT YET |
| Coverage certificates (full) | NOT YET |
| Paper-level experiment sweeps | NOT YET |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- (Optional) Docker

### Setup

```bash
# Clone / enter workspace
cd faulttrace-rag

# Backend setup
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt -r requirements-dev.txt
pip install -e packages/core -e packages/data -e packages/gold -e packages/pipelines -e packages/reporting

# Frontend setup
cd apps/web
npm install
cd ../..

# Configure environment
cp .env.example .env
```

### Run Migrations and Seed Demo

```bash
make migrate
make seed-demo
```

### Start Development Servers

```bash
make dev
```

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:3000

### Full Test Suite

```bash
make test        # all tests
make lint        # ruff + eslint
make typecheck   # mypy + tsc
make smoke       # end-to-end smoke path
```

---

## Demo Path

1. Open http://localhost:3000
2. Click **Start Demo** on the overview page
3. Navigate to **Corpus Worlds** to browse generated records
4. Open **Query Library** and pick any question
5. Click **Run Baseline** to execute the deterministic P0 pipeline
6. Navigate to **Trace Inspector** to see stage-by-stage trace

---

## Current Milestone: Prompt 1

Implements deterministic Track M data generation, procedural query factory, dual gold engine, deterministic baseline pipeline (P0-deterministic-scope-baseline), FastAPI service, and a functional Next.js research dashboard. No real LLM is used; all computation is deterministic.

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full component diagram and data flow.

## Decisions

See [docs/DECISIONS.md](docs/DECISIONS.md) for the architecture decision log.

## Build State

See [docs/BUILD_STATE.md](docs/BUILD_STATE.md) for current implementation status.
