# Related-work and novelty boundary

This matrix states the intended differentiation; it is not evidence that the contribution
is novel. A final submission requires a systematic literature search and reviewer-quality
comparison against the implemented baselines.

| Work | Established capability | Boundary for FaultTrace-RAG |
|---|---|---|
| [RAGAS (EACL 2024)](https://aclanthology.org/2024.eacl-demo.16/) | Reference-free evaluation of retrieval and generation dimensions | FaultTrace must demonstrate executable component interventions and known-fault localization, not merely another metric suite. |
| [ARES (NAACL 2024)](https://aclanthology.org/2024.naacl-long.20/) | Adaptable component-specific LLM judges for RAG evaluation | FaultTrace's oracle diagnostics require comparison with judge-based evaluation on real text; deterministic oracle access is not a substitute. |
| [RAGChecker (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html) | Fine-grained retrieval/generation diagnostic metrics with human meta-evaluation | FaultTrace targets executable structured R/E/A interventions, but must not claim that fine-grained RAG diagnosis itself is new. |
| [Source Attribution in RAG (2025 preprint)](https://arxiv.org/abs/2507.04480) | Shapley-style influence of retrieved sources | FaultTrace attributes pipeline stages R/E/A and must evaluate stage-level ground truth and compound faults. |
| [RAG-E (2026 preprint)](https://arxiv.org/abs/2601.21803/) | Gradient/Shapley analysis of retriever-generator alignment | FaultTrace must show advantages or complementary coverage for analytical extraction and aggregation failures. |
| [Stronger Baselines for RAG (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1656/) | Simple long-context baselines can match complex multi-stage RAG under controlled budgets | A real-model study must include a simple source-faithful long-context baseline and matched token budgets. |
| [Counterfactual Reasoning for RAG (ICLR 2026)](https://openreview.net/pdf/2d2cb537981d1f63aa9dba040d2e3a9daab6e0a3.pdf) | Counterfactual reasoning in RAG | The paper must distinguish diagnostic stage replacement, executable replay, and certificate evaluation from answer-level counterfactual reasoning. |
| [When Failures Propagate (2026 preprint)](https://arxiv.org/abs/2608.20627) | Injected faults, downstream re-execution, and diagnosis in agentic RAG | This is the closest overlap. FaultTrace needs a narrower analytical-RAG focus, semantic/numeric certification, a transparent exact lattice, and direct empirical comparison where feasible. |

## Defensible contribution target

FaultTrace-RAG should claim a testable combination of: (1) an explicit R/E/A analytical
pipeline abstraction; (2) component-faithful downstream replay for all eight oracle subsets;
(3) ground-truth localization evaluation under single and compound injected faults; and
(4) source-grounded fact and aggregation integrity checks showing why structural coverage
alone is insufficient. It should not claim unrestricted causal identification, universal
RAG correctness, production safety, or novelty over unimplemented comparisons.
