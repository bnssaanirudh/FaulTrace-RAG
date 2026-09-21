# Limitations

FaultTrace-RAG is a local research prototype. The current evidence is a controlled
engineering validation, not a production or general language-model evaluation.

- The authoritative attribution/structured-certification study uses synthetic Track-M
  worlds with at most 1,000 records. SciFact, HotpotQA, and COVID-QA now have separate
  retrieval and deterministic pipeline-mechanics bundles, but no learned or production-LLM
  extraction result. EDGAR, GNN, Track-T, and external-provider paths remain outside the
  reported evidence.
- The SciFact external classifier is a degenerate rule that predicts `supported` for all
  300 dev claims (accuracy 0.4133, macro-F1 0.1950). The HotpotQA evidence-sentence baseline
  has answer EM 0.00027. These results validate execution and expose failure modes; they are
  not competitive claim-verification or question-answering baselines.
- The external SciFact attribution result uses injected faults and deterministic fixtures.
  Exact-set accuracy is 0.6875 across 960 lattices, but compound conditions remain weak,
  including 0.000 for the E+A condition. This is not evidence of localization on naturally
  occurring language-model failures.
- The RAGBench certificate checks minimum lexical source-token coverage and exact numeric
  fidelity over dataset-provided responses. Its fixed threshold achieved validation
  false-certification 0.0323 but test false-certification 0.0933 at 0.3049 coverage. It is
  not a semantic-entailment or truth certificate, and the validation target did not transfer
  to the test split.
- All verified provider configurations use deterministic fixtures. P1 direct BM25 accuracy
  must not be interpreted as the performance of a production language model.
- The structured-semantic certificate establishes consistency with immutable structured
  source records and a declared aggregation program. It does not establish source truth,
  freshness, user-intent correctness, fairness, safety, or textual entailment.
- Zero observed false certification is not a mathematical zero-risk guarantee. The result
  is conditional on the implemented injections, queries, and dataset, and v2 sharply lowers
  coverage under fact and aggregation faults.
- Counterfactual attribution explains recoverable error under the specified oracle
  interventions (e.g., R/E/A). It is not causal identification of model internals.
- Exact Shapley value computation has exponential time complexity $O(2^N)$ with respect to the number of pipeline stages $N$. While the underlying execution engine generalizes to arbitrary Directed Acyclic Graphs (DAGs), it practically caps $N \\le 12$, making exact calculation unsuitable for highly granular micro-service attribution without relying on approximation methods.
- The audit uses 20 deterministically selected runs per P0–P5 pipeline. Stage-matched
  artifact diagnosis obtains 0.817 single-fault exact-set accuracy and 0.901 macro-F1, but
  requires offline oracle access and still makes substantial errors.
- Dominant-component output cannot represent compound fault sets. Stage-matched artifact
  comparison improves two- and three-fault exact-set accuracy to 0.550 and 0.350, but this
  exploratory analysis is not sealed confirmatory evidence.
- Injection-intent and realized-discrepancy labels differ when a corruption operation is
  observationally silent. The present benchmark scores against injection intent, which can
  penalize a diagnostic for correctly reporting no realized artifact change.
- H1 and H2 remain untested. No matched repair-benefit or formal scale-interaction claim is
  supported by the current bundle.
- Query-clustered bootstrap intervals address repeated runs of the same query but do not
  provide dataset-level generalization from a single synthetic generator.
- The external files are locally complete and hashed, but their historical acquisition
  commands and exact upstream revisions were not recorded. Provenance therefore begins at
  the local snapshot. HotpotQA and COVID-QA results use supplied candidate sets and must not
  be represented as open-domain retrieval.
- No authentication or multi-tenant authorization is implemented. The application is
  constrained to localhost and must not be exposed without an external security layer.
- Local latency is host-dependent and descriptive. Missing cost, extraction, and retrieval
  fields remain explicitly unavailable rather than simulated.
- The enforced mypy gate covers core and gold. Repository-wide typing is not yet a release
  gate.

The exact result and analysis hashes are listed in [results.md](results.md).
