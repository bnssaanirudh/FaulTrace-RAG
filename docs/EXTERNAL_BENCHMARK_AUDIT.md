# External benchmark audit

Audit date: 2026-08-28. Canonical measured artifacts:

- retrieval: `artifacts/verified/20260828_external_retrieval_v4_final`;
- deterministic pipeline mechanics: `artifacts/verified/20260828_external_e2e_v4_final`.

## Classification

| Local material | Classification | Evaluation decision |
|---|---|---|
| SciFact BEIR corpus, queries, and train/test qrels | Complete locally usable external retrieval snapshot | Evaluated on all 300 test-qrel queries against 5,183 documents |
| SciFact official corpus, claims, and five cross-validation folds | Complete official-format companion snapshot | Dev claims used for deterministic claim-status and injected-attribution audits; not used for the BEIR retrieval result |
| HotpotQA distractor train shards and validation shard | Complete locally usable external distractor snapshot | Evaluated on all 7,405 validation questions for supplied-candidate retrieval and deterministic evidence-sentence output |
| RAGBench COVID-QA train, validation, and test Parquet files | Complete locally usable external snapshot | Retrieval scored on 242 of 246 test rows; validation/test responses used for lexical/numeric certificate calibration and evaluation |
| `apps/api/tests/fixtures/*benchmark*_cases.json` | Small hand-authored fixtures | Used only for deterministic tests; never counted as external benchmark evidence |
| Two Springer dataframe pickle files | Unverified structural/commercial local material | Excluded: acquisition/license provenance is insufficient, and unpickling an untrusted pickle can execute code |

## Integrity checks

- Every explicit source file is SHA-256 hashed in a dataset manifest.
- Query identifiers have zero overlap across the declared splits of all three evaluated datasets.
- Every scored query is non-empty and has at least one positive relevance label.
- Every gold document is present in the scored corpus or supplied candidate set.
- Duplicate COVID-QA document texts are collapsed to one deterministic hash ID per query.
- Rankings contain no duplicate document IDs, and all reported metrics are bounded by [0,1].
- Source hashes were recomputed after evaluation and were unchanged.
- All entries in the result checksum manifest were independently verified.

## Task-semantics boundary

SciFact is a conventional full-corpus retrieval task. HotpotQA distractor and RAGBench
COVID-QA include candidate contexts with each question. Pooling those contexts across
unrelated questions would invent a different benchmark, so the runner evaluates the
supplied candidate sets independently. Their high recall at the largest k simply confirms
that all candidates are returned; nDCG and smaller-k recall express ranking quality.

The retrieval bundle measures retrieval only. It provides no evidence for answer
generation, fact extraction, R/E/A fault localization, counterfactual repair benefit, or
semantic certification.

## Deterministic pipeline-mechanics audit

The mechanics bundle contains 8,911 per-case rows and records the provider or policy used
for every task. It reports:

| Evaluation | Cases | Result |
|---|---:|---:|
| SciFact deterministic claim-status baseline | 300 | Accuracy 0.4133 [0.3567, 0.4700]; macro-F1 0.1950 |
| SciFact injected R/E/A attribution | 960 lattices | Exact-set 0.6875 [0.6583, 0.7167]; macro-F1 0.7009 |
| HotpotQA deterministic evidence-sentence baseline | 7,405 | EM 0.00027; token F1 0.06067; supporting-document R@2 0.5911 |
| RAGBench calibrated lexical/numeric certificate | 246 test rows | Coverage 0.3049; false-certification 0.0933 |

The SciFact claim rule predicted `supported` for every dev example. Structural
certification covered all SciFact examples with false-certification 0.5867. The injected
attribution audit executed all eight subsets and achieved full-oracle correctness 1.0000
with zero maximum absolute efficiency residual, but compound exact-set accuracy ranged
from 0.000 to 0.500.

The HotpotQA baseline returns one retrieved evidence sentence rather than a generated
answer. Its quote-grounding rate is 1.0000, but the all-covered structural certificate has
false-certification 0.99973 against exact match. This is deliberate negative evidence for
equating evidence presence with answer correctness.

The RAGBench threshold 0.64 was selected on 267 validation rows and then held fixed.
Validation coverage/false-certification were 0.3483/0.0323; sealed-test
coverage/false-certification were 0.3049/0.0933. The policy checks lexical source-token
coverage and exact numeric fidelity. It does not establish semantic entailment or truth.

The mechanics audit validates external data flow, deterministic intervention execution,
and a narrow source-consistency policy. It does not validate learned extraction,
production LLMs, open-ended generation, or general semantic correctness.

## Provenance boundary

The files contain Hugging Face schema metadata where applicable and match the expected
dataset structures, but the original download commands and exact upstream revisions were
not recorded. The manifests therefore claim a complete, immutable local snapshot from the
time of audit onward. They do not claim a cryptographically authenticated upstream-to-local
chain. External files remain Git-ignored and are excluded from source releases.

The mechanics checksum manifest independently validates every source manifest, the 8,911
row result file, README, and summary. Trace anchors are:

- summary SHA-256: `f30a87572b0caf1ddbf93145c895f458c8ab341749751cbd95a212a74002e75b`;
- per-case Parquet SHA-256: `2e55f64974f7f7d63f6619fec70a0e74751a8dd3b46079ca702946c826a79f21`.

The catalog uses the upstream dataset licenses: SciFact CC BY-NC 2.0, HotpotQA CC BY-SA
4.0, and RAGBench COVID-QA CC BY 4.0. Users must independently confirm that their intended
use complies with those licenses.
