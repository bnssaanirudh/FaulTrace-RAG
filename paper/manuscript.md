# FaultTrace-RAG: Interventional Fault Localization and Source-Grounded Certification for Analytical RAG

## Abstract

Analytical retrieval-augmented generation systems can return a wrong aggregate for at
least three materially different reasons: the system selected the wrong records, extracted
incorrect facts, or applied an incorrect aggregation. End-to-end accuracy reveals the
failure but does not localize it, while evidence-coverage checks may accept a derivation
whose values or arithmetic are wrong. We present FaultTrace-RAG, a research framework that
represents an analytical pipeline as retrieval/scope (R), extraction (E), and aggregation
(A), executes the complete eight-subset oracle-replacement lattice, and computes exact
Shapley recoverable-error contributions. We also distinguish structural coverage from a
source-grounded structured-semantic policy that verifies record provenance, extracted
field fidelity, and deterministic aggregation replay without reading the gold answer.
In a clean controlled evaluation comprising 1,376 synthetic Track-M pipeline runs, all
runs completed and seven of seven reproducibility bundles validated. Structural
certification exhibited false-certification rates of 0.697 for wrong-fact runs and 0.852
for wrong-aggregation runs. The structured-semantic policy reduced the observed rates to
0.000 while reducing coverage from 0.908 to 0.150 and from 0.958 to 0.142, respectively.
A balanced 120-lattice audit compared dominant Shapley attribution, thresholded
outcome-active contributions, and stage-matched oracle-artifact discrepancy diagnosis.
Artifact diagnosis achieved 0.725 exact-set accuracy (query-clustered 95% interval
[0.581, 0.854]) and 0.884 macro-F1, compared with 0.500 and 0.646 for the dominant rule
(paired randomization \(p\approx 10^{-4}\)). All full-oracle runs reached zero loss.
Two- and three-fault exact-set accuracy remained 0.550 and 0.350, respectively. These
results validate the mechanism on controlled structured data. Separate external audits
measure retrieval on SciFact, HotpotQA, and RAGBench COVID-QA and exercise deterministic
text-pipeline mechanics. Structural certification falsely certified 0.587 of covered
SciFact cases and 0.9997 of covered HotpotQA cases; a validation-calibrated lexical/numeric
policy on RAGBench achieved 0.305 test coverage at 0.093 false-certification. Because the
providers were deterministic and the RAGBench policy is lexical/numeric, these audits do
not establish production-LLM quality or general semantic correctness.

**Keywords:** retrieval-augmented generation, fault localization, counterfactual
evaluation, Shapley values, selective prediction, provenance, analytical question answering

## 1. Introduction

Retrieval-augmented generation is commonly evaluated as a single input-output system.
This abstraction is inadequate for corpus-level analytical questions such as counts,
means, comparisons, trends, and rankings. A numerically incorrect answer may result from
an incomplete or contaminated record set, an error in extracted fields, or an aggregation
that does not implement the declared operation. The same final error can therefore arise
from different repair targets.

Component metrics partially address this problem, but a static score does not answer the
interventional question: how would the final answer change if one pipeline component were
replaced while the remaining components continued to execute? This question becomes
especially important when upstream replacement changes downstream inputs. Replacing
retrieval with an oracle record set is not a valid intervention if the system merely
filters a cached extraction produced from the original retrieval. The replacement must
flow through the evaluated extractor and aggregator.

Certification creates a related problem. A trace can contain every expected record and
still be wrong because the extracted values are corrupted or the final arithmetic is
incorrect. Calling such a certificate a correctness certificate would conflate evidence
coverage with semantics.

FaultTrace-RAG addresses these problems with four scoped contributions:

1. an explicit R/E/A model for structured analytical RAG;
2. component-faithful execution of all eight oracle-replacement subsets;
3. fault-localization evaluation against known single and compound injections; and
4. a layered certificate that separates structural coverage from source-and-program
   consistency.

The claims are deliberately narrow. The framework diagnoses recoverable error under its
defined interventions. It does not identify unrestricted causal mechanisms inside a model,
prove source truth, or certify general natural-language entailment.

## 2. Related work and novelty boundary

RAGAS introduced reference-free evaluation dimensions for retrieval and generation
([Es et al., 2024](https://aclanthology.org/2024.eacl-demo.16/)). Later attribution work
uses Shapley values to quantify the influence of retrieved sources
([Source Attribution in RAG, 2025](https://arxiv.org/abs/2507.04480)), while RAG-E combines
gradient and Shapley-style analysis for retriever-generator alignment
([RAG-E, 2026](https://arxiv.org/abs/2601.21803/)). Counterfactual reasoning has also been
studied directly in RAG
([ICLR 2026 paper](https://openreview.net/pdf/2d2cb537981d1f63aa9dba040d2e3a9daab6e0a3.pdf)).
ARES trains component-specific evaluation judges
([Saad-Falcon et al., 2024](https://aclanthology.org/2024.naacl-long.20/)), while
RAGChecker reports fine-grained retrieval and generation diagnostics validated against
human judgments
([Ru et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html)).
A concurrent preprint, *When Failures Propagate*, is particularly close in its use of
injected faults and downstream re-execution
([2026 preprint](https://arxiv.org/abs/2608.20627)).

FaultTrace therefore does not claim novelty for component evaluation, counterfactual
reasoning, or Shapley attribution in isolation. Its target differentiation is the
combination of an explicit analytical R/E/A interface, exact executable component replay,
ground-truth stage-fault evaluation, and source-grounded numeric/aggregation certification.
Direct empirical comparison with the closest recent systems remains future work.

## 3. Problem formulation

Let a pipeline answer be

\[
\hat y = A_{\theta_A}(E_{\theta_E}(R_{\theta_R}(q,D))),
\]

where query \(q\) executes over corpus \(D\). R returns a record scope, E returns
structured fact rows, and A reduces those rows to an answer. For controlled evaluation,
oracle components \(R^*\), \(E^*\), and \(A^*\) are available.

For each subset \(S\subseteq\{R,E,A\}\), components in S are replaced with their
oracles and all downstream stages execute on the replacement output. Let \(L(S)\) be
the query-family-specific loss. We define

\[
v(S)=L(\varnothing)-L(S).
\]

Exact Shapley values are computed over the eight subsets. For example,

\[
\phi_R=\tfrac13[v(R)-v(\varnothing)]
+\tfrac16[v(RE)-v(E)]
+\tfrac16[v(RA)-v(A)]
+\tfrac13[v(REA)-v(EA)].
\]

The corresponding expressions define \(\phi_E\) and \(\phi_A\). Values may be
negative: an oracle replacement can expose a compensating error and worsen the answer.
Invalid lattice subsets invalidate the entire attribution rather than being assigned zero.
For a complete exact lattice, Shapley efficiency gives

\[
\phi_R+\phi_E+\phi_A=v(REA)
\]

up to numerical precision. The retained interaction field is only this numerical residual;
it is not a Shapley interaction index.

## 4. Gold construction and leakage boundary

Query specifications contain a typed scope-predicate AST, required facts, an aggregation
specification, and a comparison tolerance. Independent Pandas and DuckDB evaluators execute
each specification. A gold answer is persisted only when their recursively compared
outputs agree within tolerance. Recursive comparison covers nested lists and mappings as
well as scalar numeric values.

Gold answers are quarantined to offline scoring and declared oracle interventions. The
normal pipeline and certificate do not read them. Dataset manifests record exact files,
hashes, source metadata, and split overlap. Experiment jobs record the dataset snapshot,
world record-set hash, query-specification hash, gold-answer hash, execution seed, provider,
model label, and pipeline configuration.

## 5. Component-faithful intervention execution

The intervention runner resolves an explicit pipeline implementation for the parent run.
When R is replaced but E is not, the oracle scope is passed into the evaluated pipeline's
`replay_extraction` boundary. When E is replaced, the extraction oracle operates on the
current scope, whether original or replaced. When A is not replaced, the evaluated
pipeline's aggregation boundary runs over the current extracted rows. This prevents a
retrieval intervention from being reduced to a filter over stale cached facts.

Random fault injection is namespaced by query, execution seed, and component. Thus changing
the extraction intervention does not unintentionally change the scope or aggregation
perturbation. The same seed is preserved when the pipeline is reconstructed for lattice
execution.

## 6. Layered certification

The structural policy checks expected-scope recall and precision, extraction completeness,
required fields, ambiguity, context truncation, and operator-specific evidence conditions.
It cannot validate values or arithmetic.

For immutable structured sources, the v2 policy adds:

- provenance coverage: each extracted record identifier maps to one source record;
- source fact fidelity: required extracted cells match source cells recursively within the
  query tolerance; and
- aggregation replay: deterministic execution of the declared aggregation over persisted
  extracted rows reproduces the returned answer.

The strict decision is

\[
C_{v2}=C_{structural}\land C_{provenance}\land C_{facts}\land C_{aggregation}.
\]

Unknown semantic dimensions cause abstention. This policy certifies source-and-program
consistency, not external truth.

## 7. Controlled experimental design

### 7.1 Dataset and queries

Track-M is a deterministic synthetic analytical corpus generator. The verified run created
nested worlds at 10, 50, 200, and 1,000 records. Queries cover count, mean, proportion,
comparison, top-k, and trend families with easy, medium, and adversarial strata. Because
the same query may occur under several configurations or seeds, the query is the
statistical cluster.

### 7.2 Fault conditions

P0 uses deterministic correct R/E/A and serves as the clean control. P1 perturbs scope,
P2 corrupts facts, P3 corrupts aggregation, P4 combines scope and fact faults, and P5
combines all three. A direct BM25 path is included in some configurations but uses a
deterministic fixture; it is not treated as a real-model baseline.

### 7.3 Outcomes and uncertainty

Pipeline outcomes are raw-answer accuracy, answer coverage, selective risk, and
false-certification rate. Accuracy receives a query-clustered 95% bootstrap interval using
10,000 resamples. Attribution outcomes are exact fault-set accuracy, macro- and micro-F1,
example-F1, Hamming loss, full-oracle recovery, and efficiency residual. We report three
decision views: (i) the largest absolute Shapley component, (ii) outcome-active components
whose cross-fitted absolute Shapley value exceeds a development-fold threshold, and (iii)
stage-matched discrepancies between persisted R/E/A artifacts and their oracle outputs.
The third is an offline oracle diagnostic, not an online certificate. The balanced audit
selects the first 20 query-ID/seed pairs per P0–P5 after deterministic sorting.

## 8. Results

All 1,376 pipeline jobs completed, and all seven bundles passed checksum validation.
Table 1 reports the principal pipeline results.

| Condition | Runs | Accuracy [95% CI] | v2 coverage | v2 FCR | Structural coverage | Structural FCR |
|---|---:|---:|---:|---:|---:|---:|
| P0 clean control | 154 | 1.000 [1.000, 1.000] | 0.942 | 0.000 | 0.968 | 0.000 |
| P1 wrong scope | 154 | 0.266 [0.186, 0.353] | 0.240 | 0.000 | 0.253 | 0.000 |
| P2 wrong facts | 120 | 0.317 [0.233, 0.400] | 0.150 | 0.000 | 0.908 | 0.697 |
| P3 wrong aggregation | 120 | 0.158 [0.092, 0.225] | 0.142 | 0.000 | 0.958 | 0.852 |
| P4 scope + facts | 421 | 0.081 [0.045, 0.123] | 0.024 | 0.000 | 0.252 | 0.736 |
| P5 scope + facts + aggregation | 387 | 0.021 [0.005, 0.040] | 0.008 | 0.000 | 0.251 | 0.959 |

Structural coverage frequently certified fact and aggregation faults. Source-fact fidelity
and aggregation replay removed all observed false certifications in the controlled study,
but coverage fell sharply. The result is therefore a risk/coverage trade-off, not evidence
of a universally superior operating point.

All 120 attribution lattices were valid, had zero efficiency residual, and reached zero
loss under full replacement. Table 2 separates diagnosis of answer-loss contribution from
diagnosis of component-artifact discrepancy.

| Decision view | Exact set | Macro-F1 | Micro-F1 | Example-F1 | Hamming loss |
|---|---:|---:|---:|---:|---:|
| Dominant absolute Shapley | 0.500 | 0.646 | 0.653 | 0.685 | 0.231 |
| Cross-fitted Shapley multilabel | 0.600 | 0.746 | 0.762 | 0.740 | 0.175 |
| Stage-matched artifact discrepancy | 0.725 | 0.884 | 0.887 | 0.873 | 0.092 |

Both cross-fitting folds selected an absolute Shapley threshold of 0.001 without scoring
their calibration rows. Stage-matched artifact diagnosis improved exact-set accuracy by
0.225 over the dominant rule; its query-clustered interval was [0.581, 0.854], versus
[0.370, 0.647] for the dominant rule, with paired randomization \(p=0.00010\) using 10,000
draws. This comparison is exploratory because artifact diagnosis was added after the
original controlled audit.

The clean controls were correctly assigned no artifact fault in all 20 cases. Artifact
diagnosis achieved 0.817 single-fault exact-set accuracy (95% clustered interval
[0.714, 0.915]) and 0.901 macro-F1. For compound faults, exact-set accuracy rose from zero
under the dominant rule to 0.550 for two faults and 0.350 for three faults. Missed labels
often corresponded to injected operations that produced no observable artifact difference
for that query, so injected-fault truth and outcome-active discrepancy should not be
conflated.

### 8.1 External retrieval-only audit

We additionally audited the complete external files available in the working copy. This
evaluation is deliberately separated from the R/E/A mechanism study. SciFact uses BEIR
test qrels against a 5,183-document corpus. HotpotQA distractor and RAGBench COVID-QA
provide a candidate set with each question, so they are scored as per-question candidate
retrieval/reranking rather than pooled or open-domain retrieval. Four of 246 COVID-QA test
rows contain no positive relevance key and were excluded before scoring.

| Dataset and task | Retriever | Queries | nDCG [95% bootstrap CI] | Recall |
|---|---:|---:|---:|---:|
| SciFact full corpus | BM25 | 300 | nDCG@10 0.6408 [0.5948, 0.6866] | R@10 0.7667 |
| SciFact full corpus | all-MiniLM-L6-v2 | 300 | nDCG@10 0.6451 [0.5991, 0.6898] | R@10 0.7833 |
| SciFact full corpus | hybrid RRF | 300 | nDCG@10 0.6873 [0.6427, 0.7305] | R@10 0.8186 |
| HotpotQA distractor supplied candidates | BM25 | 7,405 | nDCG@10 0.8293 [0.8254, 0.8330] | R@10 1.0000 |
| RAGBench COVID-QA supplied candidates | BM25 | 242 | nDCG@4 0.9032 [0.8842, 0.9215] | R@4 1.0000 |

The result artifact contains 8,547 per-query rows, explicit source-file manifests, and
zero query-ID overlap across declared splits. The historical acquisition commands were
not recorded, so provenance begins with the locally hashed snapshots rather than claiming
an end-to-end authenticated download chain. These measurements exercise only retrieval;
they neither test multi-hop answer generation nor strengthen the semantic-certificate
claim beyond Track-M.

### 8.2 External deterministic pipeline-mechanics audit

A second external audit exercises deterministic claim-status prediction, evidence-sentence
extraction, injected R/E/A attribution, and a validation-calibrated source-consistency
policy. It contains 8,911 per-case rows and labels the provider and task semantics for
every result. SciFact and HotpotQA use deterministic fixtures rather than learned models;
RAGBench scores dataset-provided responses.

| Dataset and task | Cases | Main result | Secondary result |
|---|---:|---:|---:|
| SciFact official dev, deterministic claim status | 300 | Accuracy 0.4133 [0.3567, 0.4700] | Macro-F1 0.1950; evidence R@5 0.8874 |
| SciFact injected R/E/A attribution | 960 lattices | Exact-set 0.6875 [0.6583, 0.7167] | Macro-F1 0.7009; full-oracle correct 1.0000 |
| HotpotQA deterministic evidence sentence | 7,405 | Answer EM 0.00027 [0.00000, 0.00068] | Token F1 0.06067 [0.05826, 0.06310] |
| RAGBench calibrated lexical/numeric certificate | 246 test rows | Coverage 0.3049 [0.2480, 0.3618] | False-certification 0.0933 [0.0400, 0.1600] |

The SciFact rule predicted `supported` for every claim. Structural certification covered
all 300 examples but falsely certified 0.5867 [0.5300, 0.6433] of covered cases.
Injected-fault attribution executed all eight R/E/A subsets for 960 lattices with zero
maximum absolute Shapley efficiency residual. Exact-set accuracy was 1.000 for clean and
single-stage faults, 0.500 for R+E and R+A, 0.000 for E+A, and 0.500 for R+E+A.

The HotpotQA extractor returns one retrieved evidence sentence. Quote grounding was
1.000, but answer EM was effectively zero; the all-covered structural certificate had
false-certification 0.99973 against exact match. On RAGBench, threshold 0.64 was selected
on 267 validation examples (coverage 0.3483, false-certification 0.0323) and held fixed on
test, where precision was 0.9067 and recall was 0.3285. The test false-certification rate
exceeded the validation target, illustrating both the coverage cost and the calibration
transfer risk of the narrow policy.

## 9. Discussion

The certification experiment gives direct evidence for the paper's central negative
claim: completeness is not correctness. P2 and P3 preserve correct scope, so structural
coverage remains high even when the facts or reducer are wrong. The v2 checks detect these
faults without gold by comparing extraction artifacts with immutable source rows and
replaying the declared reducer.

The result demonstrates why a single diagnostic label is inadequate. Cross-fitted
thresholding recovers multiple outcome-active components, while direct artifact comparison
can identify a wrong stage even when downstream masking gives it near-zero Shapley effect.
The latter requires oracle access and therefore cannot be used as a deployment-time causal
explanation or certificate.

Compound diagnosis remains incomplete. Three-fault exact-set accuracy of 0.350 means most
complete fault sets are still missed, despite micro-F1 of 0.879 in that subgroup. The gap
also exposes a label problem: a fault injector can execute without changing an artifact,
whereas the benchmark label still records the attempted injection. Future confirmatory
evaluation should predeclare both injection-intent and realized-discrepancy labels and use
a sealed calibration split.

The external mechanics audit reinforces the paper's negative certification claim rather
than establishing a new model-quality claim. The HotpotQA baseline is perfectly grounded
as a quote but almost never answers the question exactly, while the SciFact rule is
structurally complete but degenerate as a classifier. RAGBench shows that stricter
source-consistency checks can improve precision only by abstaining on most examples and
that a validation operating point need not transfer at the target false-certification
rate.

## 10. Threats to validity

Internal validity benefits from dual-engine gold, immutable data, typed artifacts,
namespaced seeds, complete-lattice rejection, and checksummed bundles. Nevertheless,
fault injections are designed rather than naturally observed, and their severity
distribution may favor the implemented checks.

Construct validity is limited because R/E/A is one abstraction of an analytical system.
Planning, query interpretation, chunking, prompting, and source quality may constitute
additional fault stages. Source-field equality is not semantic entailment for free text.

External validity is the principal limitation for the mechanism claims. All headline
attribution and structured certification results are from one synthetic generator and
deterministic providers. The external audits show that retrieval adapters, deterministic
text-pipeline interventions, and narrow source-consistency policies operate on three real
benchmark snapshots. They cannot support claims about learned extraction, production
language models, unrestricted semantic certification, other languages, or deployment
conditions.

Statistical conclusion validity is limited by one dataset domain. Query clustering prevents
pseudo-replication across seeds, but no dataset-level replication is available.

## 11. Reproducibility and artifact integrity

The authoritative result file has SHA-256
`53f3d4974c706ed4d73c05d10b6ace02d5f33ef4427fd9632974024a717aff5c`;
the dataset manifest has SHA-256
`c253a490f99e19be73e2af140438210376348c228a8affcc0d235b7931485c29`;
the attribution file has SHA-256
`b823bea8b0b362063b23658c0ef0217e1ac8777bbe2ad7e01cf600afde4d57be`.
The full paths and secondary-analysis hash are listed in [results.md](results.md).
The external retrieval summary has SHA-256
`4dd80323ed4b065ad9d0b2e06957a992a7e219d9464a849655d1fe42627cbd8d`;
its per-query Parquet file has SHA-256
`03faaaafeb1af4f860004687c91ea5a9e0d0bb9e2be70d342f24e8af67f0bacc`.
The external mechanics summary has SHA-256
`f30a87572b0caf1ddbf93145c895f458c8ab341749751cbd95a212a74002e75b`;
its per-case Parquet file has SHA-256
`2e55f64974f7f7d63f6619fec70a0e74751a8dd3b46079ca702946c826a79f21`.
Unavailable measures remain null and are never replaced with simulated values.

## 12. Conclusion

FaultTrace-RAG provides an executable framework for asking which analytical RAG stage
accounts for recoverable answer error and whether a structured derivation is internally
consistent with its source records and aggregation program. Controlled results show why
structural evidence coverage is insufficient and demonstrate improved oracle-assisted
multilabel localization. They also reveal two substantial costs: conservative semantic
checks reduce coverage, and complete compound-fault recovery remains weak. Real datasets,
learned models, matched end-to-end baselines, independently calibrated multilabel
attribution, and a sealed confirmatory evaluation are required before journal-level
generalization. The external deterministic audits strengthen the failure analysis and
calibration evidence but do not remove those requirements.
