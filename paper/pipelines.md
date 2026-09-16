# Pipelines and intervention boundaries

The controlled fault family decomposes an analytics run into retrieval/scope (R), fact extraction (E), and aggregation (A):

- **P0 deterministic scope baseline:** correct deterministic scope, facts, and aggregation. It is an oracle control, not an LLM baseline.
- **P1 wrong scope:** perturbs the scope predicate while retaining deterministic fact handling and aggregation.
- **P2 wrong facts:** preserves scope and deterministically corrupts selected extracted fields before aggregation.
- **P3 wrong aggregation:** preserves scope and facts and corrupts the final aggregation.
- **P4 compound scope + facts:** combines P1 and P2 faults while retaining the aggregation implementation.
- **P5 full compound:** combines scope, fact, and aggregation faults.

The registry also contains experimental direct BM25, dense retrieval, extract-and-aggregate, MER repair, and GNN paths. Only P1 direct BM25 appears in the verified top-k configurations, and it uses the deterministic provider fixture in those runs. The remaining experimental paths are not covered by the regenerated aggregate claims.

Counterfactual evaluation executes all subsets of replacing R, E, and A. A replacement of R passes the replacement scope into the evaluated pipeline's `replay_extraction` boundary; a replacement of E supplies oracle facts; and replacement of A invokes deterministic oracle aggregation. Invalid or incomplete lattices are reported rather than assigned zero loss.

The resulting Shapley values quantify the measured effect of these defined replacements. They should be described as intervention-specific recoverable-error attribution, not proof of a unique real-world failure cause.
