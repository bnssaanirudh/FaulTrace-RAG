# FaultTrace-RAG

> **Compatibility Note**: Historical experiment artifacts may retain the legacy "FaulTrace" spelling. Their filenames are preserved to maintain checksum and provenance integrity.

FaultTrace-RAG is a local research prototype for diagnosing errors in corpus-level analytics pipelines. It models a run as three replaceable stages—retrieval/scope (R), fact extraction (E), and aggregation (A)—then measures how deterministic oracle replacements change the answer.

The system emits explicitly scoped certificates. Structural certificates describe evidence
membership and completeness. For immutable structured sources, v2 additionally checks
record provenance, source-field fidelity, and deterministic aggregation replay. Neither
policy is a truth, safety, or security certificate.

## What is implemented

- Deterministic Track-M corpus generation at multiple nested scales.
- Query generation for count, mean, proportion, comparison, top-k, and trend families.
- Dual Pandas/DuckDB gold evaluation; a gold answer is persisted only when both engines recursively agree within the query tolerance.
- P0 oracle evaluation, direct retrieval baselines, and controlled R/E/A fault-injection pipelines.
- Full eight-subset counterfactual replay. Replacing retrieval can rerun the pipeline's extraction stage, and aggregation replacement invokes the pipeline aggregation boundary.
- FastAPI and Next.js applications, with integration coverage for every frontend API call.
- Measured-only reporting and reproducibility bundles containing configs, metrics, environment locks, source snapshots, and checksums.
- Traceable external retrieval evaluation for SciFact, HotpotQA distractor, and RAGBench COVID-QA with explicit task-semantics boundaries.
- A separate external pipeline-mechanics audit using deterministic extractors and a validation-calibrated lexical/numeric certificate. Deterministic outputs are never labeled as LLM results.

SciFact has full-corpus retrieval validation; HotpotQA and COVID-QA have supplied-candidate
retrieval validation. A narrow external audit also exercises deterministic claim-status
prediction, evidence-sentence extraction, injected R/E/A attribution, and source-consistency
certification. It does not establish learned-extractor, production-LLM, semantic-entailment,
or truth-certification performance. GNN, EDGAR, and external-provider paths remain
experimental integrations.

## Security boundary

There is no user authentication or multi-tenant authorization. The API and Docker ports bind to `127.0.0.1` by default, provider-connectivity tests are disabled by default, and ingestion is restricted to configured trusted roots.

**Do not expose this application to a LAN or the public internet.** Add authentication, authorization, TLS termination, and deployment-specific hardening before any non-local deployment.

## Installation and local use

Requirements: Python 3.11+ and Node.js 20+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX: source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e packages/core -e packages/data -e packages/gold \
  -e packages/pipelines -e packages/reporting -e apps/api

cd apps/web
npm ci
cd ../..

python -m alembic -c alembic.ini upgrade head
python -m faulttrace_api.bootstrap
```

Start the API and web application in separate terminals:

```bash
python -m uvicorn faulttrace_api.main:app --host 127.0.0.1 --port 8000 --app-dir apps/api
cd apps/web && npm run dev
```

- Web: <http://localhost:3000>
- API: <http://localhost:8000>
- OpenAPI UI: <http://localhost:8000/docs>

On Windows, the equivalent shortcuts are `make setup`, `make seed-demo`, and `make dev`.

### Docker

```bash
docker compose up --build
```

Compose initializes the database, seeds the demo idempotently, and binds both published ports to localhost. The current repository configuration validates with `docker compose config`; an image build additionally requires a running Docker daemon.

## Verification

```bash
python -m pytest apps/api/tests packages --basetemp=.tmp_pytest -p no:cacheprovider
python -m ruff check packages apps/api scripts
python -m mypy packages/core/faulttrace_core packages/gold/faulttrace_gold \
  --ignore-missing-imports --no-site-packages
cd apps/web && npm run lint && npm run type-check && npm run build
```

The enforced Python typing gate currently covers core and gold (24 modules). Repository-wide mypy is not yet a release gate; see the documented typing limitation below.

Regenerate every checked-in experiment from a new immutable dataset directory:

```bash
python scripts/run_verified_experiments.py --output-dir artifacts/verified/<new-run-id>
```

The runner refuses to reuse an existing directory, fingerprints and read-protects the generated dataset, verifies that it did not change, records per-job dataset/query/gold/source provenance, and fails when an underlying pipeline run fails.

## Research Provenance

> Experimental results were generated against commit `433f3d580e0fe526113e94af89398d137ffc84a0`. Later repository commits may contain documentation, archival packaging, or presentation updates and must not be described as having generated those measurements unless experiments are rerun.

## Evolution of the research framework

The FaultTrace-RAG research program has evolved from a hard-coded three-stage model into a generalized multi-stage evaluation and repair framework. The analyses are structured chronologically:

### Phase A — deterministic analytical framework
*(Confirmatory)*
The original formulation models a pipeline as three replaceable stages: retrieval/scope (R), fact extraction (E), and aggregation (A). By re-executing downstream stages after injecting an oracle replacement, it computes exact Shapley and active contributions to localize recoverable error.

### Phase B — controlled compound-fault localization
*(Engineering Validation)*
Controlled synthetic experiments on the Track-M dataset validated the deterministic mechanisms, establishing baseline metrics for artifact-discrepancy diagnosis under compound fault injections.

### Phase C — generalized multi-stage counterfactual diagnosis
*(Theoretical Extension)*
The R/E/A formulation is generalized to an arbitrary directed acyclic component graph (e.g., an extended five-stage S → R → E → A → G graph), defining counterfactual interventions over any pipeline stage.

### Phase D — active diagnosis and repair
*(Methodological)*
Exhaustive counterfactual enumeration requires \(2^n\) runs. Bayesian Active Counterfactual Diagnosis (BACD) and Minimum Counterfactual Repair (MCR) optimize this by selecting cost-aware probes to localize and repair faults within budgets.

### Phase E — external/live-model validation
*(External Validation)*
The framework was evaluated on external datasets (SciFact, HotpotQA, COVID-QA, RAGTruth) and on 200 audited natural-failure cases from live LLMs (Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3), evaluating real-world diagnosis and repairability. (Preliminary Phi-4 experiments were excluded.)

### Phase F — certification limitations
*(Exploratory & Negative Findings)*
While empirical structural certificates achieved zero observed false-certification in specific environments, stricter statistical risk-controlled sweeps (e.g., RAGTruth) yielded zero held-out certified coverage under the tested settings. The results confirm that current source-grounding checks are structural constraints, not universal truth guarantees.

## Generated artifacts and releases

Generated datasets, databases, plots, bundles, frontend build output, and caches are ignored and excluded from source releases. See [the generated-artifact policy](docs/GENERATED_ARTIFACTS.md).

```bash
python scripts/export_openapi.py
python scripts/build_release.py
```

The release builder creates a source-only zip plus a SHA-256 manifest under `dist/`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Installation](INSTALLATION.md)
- [Build state](docs/BUILD_STATE.md)
- [External benchmark audit](docs/EXTERNAL_BENCHMARK_AUDIT.md)
- [Paper results](paper/results.md)
- [Current manuscript](paper/manuscript.md)
- [Confirmatory protocol](paper/preregistration.md)
- [Novelty boundary](paper/novelty_matrix.md)
- [Limitations](paper/limitations.md)

## License

MIT. See [LICENSE](LICENSE).
