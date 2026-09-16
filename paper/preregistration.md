# FaultTrace-RAG confirmatory evaluation protocol

Status: protocol for future external-dataset/model experiments. The deterministic Track-M
study is an engineering validation and must not be relabeled as this confirmatory study.

## Research questions and hypotheses

- **H1 (scale interaction):** fault impact changes with corpus scale after controlling for
  query family and difficulty.
- **H2 (matched repair benefit):** a repair pipeline reduces paired query-level loss or
  selective risk relative to the same pipeline without repair at comparable measured cost.
- **H3 (single-fault localization):** for runs with exactly one injected R, E, or A fault,
  the largest absolute Shapley contribution identifies that component more often than the
  strongest non-counterfactual baseline.
- **H4 (compound-fault localization):** thresholded component contributions improve
  macro-F1 over dominant-component-only attribution for compound faults.
- **H5 (semantic certification):** the structured-semantic policy reduces false
  certification relative to the structural policy on the same runs, while retaining
  non-zero answer coverage on clean runs.

H3 and H5 are primary. H1, H2, and H4 are secondary until the required matched designs
exist. Hypotheses will not be rewritten after inspecting the sealed test results.

## Experimental units and independence

The query is the statistical cluster. Model samples, seeds, retrieval configurations, and
pipeline variants for the same query are repeated measurements rather than independent
observations. Dataset is treated as a replication domain, not as a row-level sample.

## Required domains

The confirmatory study requires at least one controlled synthetic analytical dataset, two
real analytical or numerical datasets, and one multi-hop retrieval dataset. Dataset files,
licenses, source URLs, transformations, and split hashes must be captured before execution.
No result from a fixture-sized adapter test may appear in a primary empirical table.

## Required systems

The design requires BM25, dense, hybrid, and oracle retrieval; at least three real language
models from more than one model family; and structural, semantic, and hybrid certification
policies. Exact model revisions, decoding parameters, provider dates, prompts, and token
accounting must be frozen in the experiment manifest.

## Outcomes

Primary outcomes are single-fault top-1 localization accuracy and false-certification rate.
Secondary outcomes are macro-F1, exact fault-set accuracy, selective risk, answer coverage,
risk-coverage area, full-oracle recovery loss, calibration error, latency, token use, and
measured financial cost.

## Statistical analysis

- Report query-clustered 95% bootstrap confidence intervals using 10,000 resamples.
- Use paired permutation tests for matched systems.
- Report absolute differences and standardized effect sizes.
- Apply Holm correction within each hypothesis family.
- Treat dataset-level consistency as necessary evidence; do not rely only on pooled rows.
- Exclude a run only for a predeclared infrastructure failure. Report all exclusions and
  count invalid counterfactual lattices as failures, not zeros.

## Leakage and stopping rules

Development uses training/development splits. Test splits remain sealed until code,
thresholds, prompts, and analysis scripts are frozen. Data collection ends at the declared
matrix size; significance is not used as an early-stopping criterion. Gold answers and
oracle outputs may be used only by offline evaluation and declared interventions, never by
the normal certificate or generation path.

## Reporting rules

All tables and plots must be generated from immutable bundles. Missing metrics are reported
as unavailable. Estimated planning costs, fixture outputs, and simulated UI data are not
empirical results. Negative, null, and contradictory results are retained.
