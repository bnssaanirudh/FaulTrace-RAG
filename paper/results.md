# Regenerated controlled engineering-validation results

## Provenance

All values on this page come from the clean immutable execution at
`artifacts/verified/20260828_q1_multilabel_v7_final`. The runner created nested
synthetic Track-M worlds at N = 10, 50, 200, and 1,000, persisted only
dual-engine-agreed gold answers, made the generated dataset read-only, executed every
checked-in configuration, and verified every bundle checksum.

- Measurement status: `measured_from_clean_local_execution`
- Configurations with valid bundles: 7 / 7
- Completed pipeline runs: 1,376 / 1,376
- Failed pipeline runs: 0
- Valid attribution lattices: 120 / 120
- Result SHA-256: `53f3d4974c706ed4d73c05d10b6ace02d5f33ef4427fd9632974024a717aff5c`
- Dataset-manifest SHA-256: `c253a490f99e19be73e2af140438210376348c228a8affcc0d235b7931485c29`
- Attribution SHA-256: `b823bea8b0b362063b23658c0ef0217e1ac8777bbe2ad7e01cf600afde4d57be`
- Attribution-metrics SHA-256: `ebad459aaaffe1bec01be20b7532d705cb66b9eccc48ba3dded1d99f491eb811`
- Secondary-analysis SHA-256: `0a960985b7382e4bbff6d45cac5854fd11397b7f55d139261558dabce0e78f99`
- Executable tracked-diff SHA-256: `46debdf51d6751ba18c5dca8c4cdb42999258d44187fcec0fbcfe3a75071f787`
- Git HEAD: `433f3d580e0fe526113e94af89398d137ffc84a0`

## Pipeline accuracy and certification

Accuracy intervals are query-clustered 95% bootstrap intervals with 10,000
resamples. Repeated configurations or seeds for the same query move together. The
structural-policy columns are counterfactual decisions obtained by removing only the
v2 semantic blockers from the same certificates; gold is used afterward to score false
certification, never by either policy.

| Pipeline | Runs | Accuracy [95% CI] | v2 coverage | v2 false-certification | Structural coverage | Structural false-certification |
|---|---:|---:|---:|---:|---:|---:|
| P0 deterministic oracle | 154 | 1.000 [1.000, 1.000] | 0.942 | 0.000 | 0.968 | 0.000 |
| P1 direct BM25 + deterministic fixture | 20 | 0.000 [0.000, 0.000] | 0.000 | n/a | 0.000 | n/a |
| P1 wrong scope | 154 | 0.266 [0.186, 0.353] | 0.240 | 0.000 | 0.253 | 0.000 |
| P2 wrong facts | 120 | 0.317 [0.233, 0.400] | 0.150 | 0.000 | 0.908 | 0.697 |
| P3 wrong aggregation | 120 | 0.158 [0.092, 0.225] | 0.142 | 0.000 | 0.958 | 0.852 |
| P4 wrong scope + facts | 421 | 0.081 [0.045, 0.123] | 0.024 | 0.000 | 0.252 | 0.736 |
| P5 wrong scope + facts + aggregation | 387 | 0.021 [0.005, 0.040] | 0.008 | 0.000 | 0.251 | 0.959 |

The v2 policy eliminated observed false certification in this controlled run, but it
did so conservatively. On P2 and P3, answer coverage fell from 90.8% to 15.0% and from
95.8% to 14.2%, respectively. These are empirical operating points, not guarantees.

## Attribution audit

The audit selected 20 runs from each of P0–P5 after deterministic query-ID/seed sorting.
Every audit executed all eight subsets of R/E/A replacement.

- Dominant-Shapley exact-set accuracy / macro-F1: 0.500 / 0.646.
- Cross-fitted Shapley multilabel exact-set accuracy / macro-F1: 0.600 / 0.746.
- Stage-matched artifact-discrepancy exact-set accuracy: 0.725, query-clustered 95% CI
  [0.581, 0.854].
- Stage-matched artifact-discrepancy macro-F1 / micro-F1: 0.884 / 0.887.
- Paired exact-set difference over dominant attribution: +0.225, randomization
  p = 0.00010 with 10,000 draws.
- Artifact-discrepancy single-fault exact-set accuracy: 0.817, query-clustered 95% CI
  [0.714, 0.915]; macro-F1: 0.901.
- Artifact-discrepancy two-fault / three-fault exact-set accuracy: 0.550 / 0.350.
- Clean-control no-fault identification: 1.000 (20 / 20) for all three decision views.
- Full-oracle zero-loss rate: 1.000 (120 / 120).
- Mean full-oracle loss and mean absolute Shapley efficiency residual: 0.000.

Dominant-component attribution still has zero compound exact-set accuracy by construction.
The stage-matched diagnostic improves compound recovery by comparing each persisted
component artifact with an oracle evaluated on the same stage input. It is an offline
oracle diagnostic, not an online certificate. Attempted injections that leave the artifact
unchanged remain counted as false negatives against injection-intent labels; therefore the
0.550 and 0.350 compound results are useful but not a complete solution.

## Interpretation boundary

These measurements validate deterministic execution, seeded fault injection, source-grounded
structured certification, and intervention-specific attribution on synthetic Track-M. They
do not establish production-LLM performance, external-dataset generalization of the R/E/A
mechanism, natural-language entailment certification, or unrestricted causal
identification. Direct BM25 in Track-M used a deterministic provider fixture and is not a
meaningful model-quality baseline.

## External retrieval-only audit

Canonical artifact: `artifacts/verified/20260828_external_retrieval_v4_final`.
This clean run uses the complete local external files, not hand-authored fixtures.

| Dataset/task | Retriever | Queries | nDCG [95% bootstrap CI] | Recall |
|---|---:|---:|---:|---:|
| SciFact, 5,183-document corpus | BM25 | 300 | nDCG@10 0.6408 [0.5948, 0.6866] | R@10 0.7667 |
| SciFact, 5,183-document corpus | all-MiniLM-L6-v2 | 300 | nDCG@10 0.6451 [0.5991, 0.6898] | R@10 0.7833 |
| SciFact, 5,183-document corpus | hybrid RRF | 300 | nDCG@10 0.6873 [0.6427, 0.7305] | R@10 0.8186 |
| HotpotQA distractor, supplied candidates | BM25 | 7,405 | nDCG@10 0.8293 [0.8254, 0.8330] | R@10 1.0000 |
| RAGBench COVID-QA, supplied candidates | BM25 | 242 | nDCG@4 0.9032 [0.8842, 0.9215] | R@4 1.0000 |

HotpotQA and COVID-QA are per-question candidate tasks and are not comparable to
open-domain corpus retrieval. Four unlabeled COVID-QA test rows were excluded. All three
dataset manifests report zero query-ID overlap between declared splits. Independent
post-run validation found no duplicate ranked IDs, no metric outside [0,1], and no
checksum mismatch across 8,547 per-query result rows.

- External summary SHA-256: `4dd80323ed4b065ad9d0b2e06957a992a7e219d9464a849655d1fe42627cbd8d`
- Per-query Parquet SHA-256: `03faaaafeb1af4f860004687c91ea5a9e0d0bb9e2be70d342f24e8af67f0bacc`
- SciFact snapshot SHA-256: `0d49c220aa54cbcd320c9d9054ed81b2d58f170776a7f270b06cd8b46c0e83d4`
- HotpotQA snapshot SHA-256: `9a8a3bb2c682e6eadc7b2780cf58bb5094494360896345f4e4a3a3a4bcebae2a`
- RAGBench COVID-QA snapshot SHA-256: `a1dcc76fd4883515c68208e715f4f0488ca4369957e110cbcaf0f52c61134700`

Claim boundary: this is retrieval-only validation. It does not evaluate extraction,
answer generation, R/E/A attribution, or semantic certification on external data.

## External deterministic pipeline-mechanics audit

Canonical artifact: `artifacts/verified/20260828_external_e2e_v4_final`. This audit is
separate from both the synthetic Track-M study and the external retrieval-only bundle.
It uses deterministic providers for SciFact and HotpotQA and a dataset-provided response
for RAGBench. No result is a learned-extractor or production-LLM measurement.

| Dataset and task | Cases | Main result | Secondary result |
|---|---:|---:|---:|
| SciFact official dev, deterministic claim status | 300 | Accuracy 0.4133 [0.3567, 0.4700] | Macro-F1 0.1950; evidence R@5 0.8874 |
| SciFact injected R/E/A attribution | 960 lattices | Exact-set 0.6875 [0.6583, 0.7167] | Macro-F1 0.7009; full-oracle correct 1.0000 |
| HotpotQA, deterministic evidence sentence | 7,405 | Answer EM 0.00027 [0.00000, 0.00068] | Token F1 0.06067 [0.05826, 0.06310] |
| RAGBench COVID-QA, calibrated lexical/numeric certificate | 246 test rows | Coverage 0.3049 [0.2480, 0.3618] | False-certification 0.0933 [0.0400, 0.1600] |

The SciFact rule predicted `supported` for all 300 claims. Its structural certificate
covered every example and had a false-certification rate of 0.5867 [0.5300, 0.6433].
Injected-fault attribution evaluated all eight R/E/A subsets for 960 lattices, reached
zero maximum absolute Shapley efficiency residual, and recovered the full-oracle answer
in every lattice. Exact-set accuracy remained condition-dependent: 1.000 for clean and
single-stage faults, 0.500 for R+E, 0.500 for R+A, 0.000 for E+A, and 0.500 for R+E+A.

The HotpotQA extractor returns one retrieved evidence sentence rather than generating an
answer. Quote grounding was 1.000, but answer EM was effectively zero. Its structural
certificate therefore covered every example while falsely certifying 0.99973 of covered
cases against exact match. This is a direct external demonstration that evidence presence
does not imply answer correctness.

For RAGBench, the lexical/numeric certificate threshold was selected once on 267
validation rows to target validation false-certification at or below 0.05. The selected
threshold was 0.64, with validation coverage 0.3483 and false-certification 0.0323. Held
fixed on 246 test rows, it achieved precision 0.9067, recall 0.3285, coverage 0.3049, and
false-certification 0.0933. The all-covered structural baseline had false-certification
0.1585. This is a narrow source-consistency policy, not semantic entailment or truth
certification.

- Mechanics summary SHA-256: `f30a87572b0caf1ddbf93145c895f458c8ab341749751cbd95a212a74002e75b`
- Per-case Parquet SHA-256: `2e55f64974f7f7d63f6619fec70a0e74751a8dd3b46079ca702946c826a79f21`

Claim boundary: these measurements validate external pipeline plumbing, deterministic
fault intervention, and narrow source-consistency behavior. They do not establish
production-LLM quality, learned extraction, open-ended answer generation, general semantic
correctness, or dataset-level generalization of the Track-M headline results.
