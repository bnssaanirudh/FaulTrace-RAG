# Architecture Decision Log: FaultTrace-RAG

## ADR-001: SQLite + DuckDB Dual Persistence

**Date**: 2026-07-14  
**Status**: Accepted

### Context
We need both transactional app metadata (queries, runs, traces) and high-performance analytical queries over large Parquet-based corpus data.

### Decision
- **SQLite** for app metadata: worlds, queries, pipeline runs, trace events. Zero-dependency, embedded, ACID-compliant, sufficient for lab-scale concurrent access.
- **DuckDB** for benchmark analytics: reading Parquet files, executing gold evaluation SQL, and running experiment aggregations. DuckDB is file-based and columnar; it reads Parquet natively without a separate ETL step.
- **Parquet + JSONL** for corpus storage: Parquet for columnar access; JSONL for streaming and debugging.

### Consequences
- No database server required for default operation.
- Both engines support optional upgrade paths (PostgreSQL via `DATABASE_URL`; remote DuckDB catalog).
- Isolation: DuckDB never writes to the SQLite app database; SQLite never reads corpus Parquet directly.

---

## ADR-002: Safe Query DSL via Declarative Predicate AST

**Date**: 2026-07-14  
**Status**: Accepted

### Context
Pipeline evaluation requires filtering corpus records by category, date, rating, brand, and other fields. Naive approaches use f-string SQL or `eval()` on query strings, creating injection and reproducibility risks.

### Decision
Implement `ScopePredicate` as a closed, typed AST supporting:
- `EqPredicate`, `NeqPredicate`, `InPredicate`, `RangePredicate`
- `IsNullPredicate`, `IsNotNullPredicate`
- `AndPredicate`, `OrPredicate`
- No string expressions; no arbitrary code; no eval

Compile predicates to:
- Pandas boolean masks (via `PredicateCompiler.to_pandas_mask`)
- DuckDB SQL WHERE clauses (via `PredicateCompiler.to_duckdb_sql`)

### Consequences
- Gold engines share the same predicate specification but use independent code paths.
- Predicates are serializable to JSON (Pydantic v2 discriminated union).
- Invalid predicate shapes are caught at query generation time.

---

## ADR-003: Dual Gold Engine is Mandatory

**Date**: 2026-07-14  
**Status**: Accepted

### Context
The benchmark compares pipeline answers to ground truth. A single gold evaluator could have bugs that systematically agree with a wrong pipeline, producing false positives.

### Decision
Every benchmark query must have its GoldAnswer computed independently by:
1. **PandasEvaluator**: pure Python, columnar, exact arithmetic
2. **DuckDBEvaluator**: SQL-based, independent aggregation logic

A GoldAnswer is only certified when both evaluators agree within the declared tolerance. Disagreements are stored as `ValidationFailure` artifacts and excluded from the benchmark.

### Consequences
- Doubles gold computation cost (acceptable: gold is computed once, offline).
- Disagreements surface bugs in the gold engine itself, which is valuable.
- Tolerance-aware comparison handles floating-point differences between engines.

---

## ADR-004: Model Providers are Abstracted Behind an Interface

**Date**: 2026-07-14  
**Status**: Accepted

### Context
Prompt 1 uses no LLM. Prompts 2-4 will add direct-context, BM25, dense-retrieval, and oracle pipelines that call real LLMs. The system must not be rewritten each time a new provider is added.

### Decision
Define a `ModelProvider` abstract interface with:
- `generate(prompt: str, config: ProviderConfig) -> ModelOutput`
- `estimate_tokens(text: str) -> int`
- Registration via a provider registry keyed by `provider_id`

Implemented providers in Prompt 1:
- `DeterministicProvider`: returns hardcoded answers based on query hash; used for tests and demo without LLM.

### Consequences
- All pipelines depend on `ModelProvider` interface, not concrete implementations.
- Adding a new LLM (local Ollama, OpenAI-compatible) requires implementing the interface only.
- Token estimates are tracked per run for cost attribution.

---

## ADR-005: Monorepo with Pip-Editable Installs

**Date**: 2026-07-14  
**Status**: Accepted

### Context
Five Python packages (`core`, `data`, `gold`, `pipelines`, `reporting`) and one FastAPI app need to share code without a full build system.

### Decision
Use `pip install -e packages/core packages/data packages/gold packages/pipelines packages/reporting` during setup. Each package has its own `pyproject.toml`. A root `pyproject.toml` holds linting and test configuration.

### Consequences
- Simple setup; no Nx, Turborepo, or Bazel required.
- Packages are importable as `faulttrace_core`, `faulttrace_data`, etc.
- Future: can add a build pipeline (e.g., `uv workspace`) when complexity justifies it.

---

## ADR-006: Next.js App Router with TypeScript and Tailwind

**Date**: 2026-07-14  
**Status**: Accepted

### Context
The dashboard must be a professional research tool, not an admin template. It requires real-time data, server components for initial load, and client-side interactivity.

### Decision
- **Next.js 14+ App Router**: server components for initial data, client components for interactivity.
- **TypeScript strict mode**: catches integration bugs at compile time.
- **Tailwind CSS**: utility-first, consistent design tokens.
- **TanStack Query**: caching, invalidation, loading states, error handling.
- **shadcn/ui primitives**: accessible, composable components.

### Consequences
- Frontend typechecks against manually synchronized API schemas (typed API client).
- No fabricated data: all numbers sourced from FastAPI backend.
- Dark mode via Tailwind `dark:` variants.
