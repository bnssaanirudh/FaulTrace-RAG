# FaultTrace-RAG

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

## Regenerated controlled results (2026-08-28)

The authoritative local engineering-validation bundle is
`artifacts/verified/20260828_q1_multilabel_v7_final`. It is not a general model-quality
study: all configured providers were deterministic fixtures, and the data were synthetic
Track-M worlds.

| Pipeline | Runs | Accuracy | v2 coverage | v2 false-certification | Structural false-certification |
|---|---:|---:|---:|---:|---:|
| P0 deterministic oracle | 154 | 100.00% | 94.16% | 0.00% | 0.00% |
| P1 direct BM25 + deterministic fixture | 20 | 0.00% | 0.00% | n/a | n/a |
| P1 wrong scope | 154 | 26.62% | 24.03% | 0.00% | 0.00% |
| P2 wrong facts | 120 | 31.67% | 15.00% | 0.00% | 69.72% |
| P3 wrong aggregation | 120 | 15.83% | 14.17% | 0.00% | 85.22% |
| P4 wrong scope + facts | 421 | 8.08% | 2.38% | 0.00% | 73.58% |
| P5 wrong scope + facts + aggregation | 387 | 2.07% | 0.78% | 0.00% | 95.88% |

All 1,376 pipeline jobs completed, all seven bundles validated, and all 120 attribution
lattices were valid. Stage-matched artifact diagnosis achieved 72.50% overall exact-set
accuracy (query-clustered 95% CI 58.12–85.42%) and 88.38% macro-F1, versus 50.00% and
64.57% for the dominant-Shapley rule (paired randomization p = 0.00010). Clean-control
specificity and full-oracle zero-loss rate were 100%. Two- and three-fault exact-set
accuracy improved to 55.00% and 35.00%, so compound diagnosis remains incomplete.

Trace anchors:

- result SHA-256: `53f3d4974c706ed4d73c05d10b6ace02d5f33ef4427fd9632974024a717aff5c`
- dataset-manifest SHA-256: `c253a490f99e19be73e2af140438210376348c228a8affcc0d235b7931485c29`
- attribution SHA-256: `b823bea8b0b362063b23658c0ef0217e1ac8777bbe2ad7e01cf600afde4d57be`
- secondary-analysis SHA-256: `0a960985b7382e4bbff6d45cac5854fd11397b7f55d139261558dabce0e78f99`
- executable tracked-diff SHA-256: `46debdf51d6751ba18c5dca8c4cdb42999258d44187fcec0fbcfe3a75071f787`

See [the full results](paper/results.md), [manuscript](paper/manuscript.md), and
[limitations](paper/limitations.md). Zero observed false certification is a controlled
measurement, not a zero-risk guarantee; v2 achieved it with a substantial coverage cost.

## External retrieval-only results (2026-08-28)

The canonical external bundle is
`artifacts/verified/20260828_external_retrieval_v4_final`. It uses the complete local
benchmark files rather than test fixtures, records every source-file hash, reports zero
query-ID overlap across declared splits, and contains 8,547 per-query result rows.

| Dataset/task | Retriever | Queries | nDCG | Recall |
|---|---:|---:|---:|---:|
| SciFact BEIR, 5,183-document corpus | BM25 | 300 | nDCG@10 0.6408 | R@10 0.7667 |
| SciFact BEIR, 5,183-document corpus | all-MiniLM-L6-v2 | 300 | nDCG@10 0.6451 | R@10 0.7833 |
| SciFact BEIR, 5,183-document corpus | hybrid RRF | 300 | nDCG@10 0.6873 | R@10 0.8186 |
| HotpotQA distractor, supplied candidates | BM25 | 7,405 | nDCG@10 0.8293 | R@10 1.0000 |
| RAGBench COVID-QA, supplied candidates | BM25 | 242 | nDCG@4 0.9032 | R@4 1.0000 |

HotpotQA and COVID-QA are per-question candidate retrieval/reranking tasks, not
open-domain corpus retrieval. Four COVID-QA test rows without any positive relevance
label were excluded before scoring. These results validate retrieval only: they do not
test answer generation, extraction, R/E/A localization, or semantic certification.

Trace anchors:

- external summary SHA-256: `4dd80323ed4b065ad9d0b2e06957a992a7e219d9464a849655d1fe42627cbd8d`
- per-query Parquet SHA-256: `03faaaafeb1af4f860004687c91ea5a9e0d0bb9e2be70d342f24e8af67f0bacc`
- SciFact snapshot SHA-256: `0d49c220aa54cbcd320c9d9054ed81b2d58f170776a7f270b06cd8b46c0e83d4`
- HotpotQA snapshot SHA-256: `9a8a3bb2c682e6eadc7b2780cf58bb5094494360896345f4e4a3a3a4bcebae2a`
- RAGBench COVID-QA snapshot SHA-256: `a1dcc76fd4883515c68208e715f4f0488ca4369957e110cbcaf0f52c61134700`

Reproduce into a new directory with:

```bash
python scripts/run_external_retrieval_benchmarks.py \
  --output artifacts/verified/<new-external-run-id>
```

External dataset files and generated result bundles are intentionally ignored and are
not included in source releases; see [the data catalog](docs/DATA_CATALOG.md).

## External deterministic pipeline-mechanics audit (2026-08-28)

The canonical mechanics bundle is
`artifacts/verified/20260828_external_e2e_v4_final`. It uses complete locally hashed
snapshots, records 8,911 per-case rows, and labels every provider and policy explicitly.

| Evaluation | Cases | Primary result | Additional result |
|---|---:|---:|---:|
| SciFact official dev, deterministic claim-status baseline | 300 | Accuracy 0.4133 [0.3567, 0.4700] | Macro-F1 0.1950; evidence R@5 0.8874 |
| SciFact injected R/E/A attribution | 960 lattices | Exact-set 0.6875 [0.6583, 0.7167] | Macro-F1 0.7009; full-oracle recovery 1.0000 |
| HotpotQA, deterministic evidence-sentence baseline | 7,405 | Answer EM 0.0003 | Token F1 0.0607; supporting-document R@2 0.5911 |
| RAGBench COVID-QA, calibrated lexical/numeric certificate | 246 test rows | Coverage 0.3049 [0.2480, 0.3618] | False-certification 0.0933 [0.0400, 0.1600] |

The SciFact classifier predicted `supported` for all 300 claims, so its accuracy is a
degenerate deterministic baseline rather than a competitive claim-verification result.
Structural certification covered every SciFact and HotpotQA example but falsely certified
0.5867 and 0.9997 of those covered cases, respectively. On RAGBench, a threshold of 0.64
was selected on 267 validation examples (coverage 0.3483, false-certification 0.0323), then
held fixed for the sealed test result above. This policy checks lexical source coverage and
exact numeric fidelity; it does not check general semantic entailment or source truth.

Trace anchors:

- mechanics summary SHA-256: `f30a87572b0caf1ddbf93145c895f458c8ab341749751cbd95a212a74002e75b`
- per-case Parquet SHA-256: `2e55f64974f7f7d63f6619fec70a0e74751a8dd3b46079ca702946c826a79f21`

Reproduce into a new directory with:

```bash
python scripts/run_external_end_to_end.py \
  --output artifacts/verified/<new-external-e2e-run-id>
```

The runner refuses to reuse a non-empty output directory and verifies that source files
remain unchanged during execution.

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
