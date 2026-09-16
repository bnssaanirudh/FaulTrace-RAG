# Data Catalog — FaultTrace-RAG

## Overview

This document lists all datasets available in FaultTrace-RAG, their licenses, how to obtain them, and their key statistics.

---

## Track M — Amazon Reviews (Synthetic)

| Field | Value |
|---|---|
| **Type** | Synthetic, procedurally generated |
| **Generator** | `faulttrace_data.generators.track_m.TrackMGenerator` |
| **Records** | 10 / 50 / 200 / 1000 (configurable scales) |
| **Format** | Parquet + JSONL |
| **License** | MIT (generated, not derived from Amazon data) |
| **Location** | `data/generated/worlds/` |
| **Reproducibility** | Fully deterministic from `seed` parameter |

**To generate:**
```bash
python -m faulttrace_data.cli seed --seed 42 --scales 10,50,200,1000
```

---

## SciFact

| Field | Value |
|---|---|
| **Type** | Scientific claim verification benchmark |
| **Source** | Wadden et al., EMNLP 2020 |
| **URL** | https://github.com/allenai/scifact |
| **Local snapshot** | 5,183 documents; 1,109 queries; 300 scored test queries |
| **Format** | JSONL (claims + corpus) |
| **License** | CC BY-NC 2.0 (dataset; not the repository code license) |
| **Location** | `data/benchmarks/scifact/beir/scifact/` |
| **Retrieval labels** | BEIR train/test qrels |

**Usage notes:**
- `SciFactAdapter` in `packages/data/faulttrace_data/benchmarks/scifact.py` handles loading
- Corpus contains PubMed abstracts; claims are manually annotated
- The canonical retrieval run uses BEIR test qrels; the original SciFact unlabeled challenge test set is a different split.
- The official dev claims are also used by the deterministic claim-status and injected
  R/E/A attribution audit. That audit uses a rule-based provider and must not be reported
  as learned or LLM claim verification.
- Snapshot hash: `0d49c220aa54cbcd320c9d9054ed81b2d58f170776a7f270b06cd8b46c0e83d4`

---

## HotpotQA

| Field | Value |
|---|---|
| **Type** | Multi-hop question answering |
| **Source** | Yang et al., EMNLP 2018 |
| **URL** | https://hotpotqa.github.io/ |
| **Local snapshot** | 90,447 train rows; 7,405 validation rows |
| **Format** | Hugging Face Parquet |
| **License** | CC BY-SA 4.0 |
| **Location** | `data/benchmarks/hotpotqa/distractor/` |
| **Labels** | Answer string + supporting facts |

**Usage notes:**
- The measured task ranks the supplied distractor contexts per validation question. It is not fullwiki/open-domain retrieval.
- A deterministic evidence-sentence baseline also runs on all 7,405 validation questions
  for pipeline-mechanics and negative certification analysis; it is not a trained QA model.
- The `HotpotQAAdapter` in `packages/data/faulttrace_data/benchmarks/hotpotqa.py` handles loading
- Snapshot hash: `9a8a3bb2c682e6eadc7b2780cf58bb5094494360896345f4e4a3a3a4bcebae2a`

---

## COVID-QA

| Field | Value |
|---|---|
| **Type** | RAGBench COVID-QA supplied-context benchmark |
| **Source** | Galileo RAGBench |
| **URL** | https://huggingface.co/datasets/galileo-ai/ragbench |
| **Local snapshot** | 1,252 train; 267 validation; 246 test rows |
| **Format** | Hugging Face Parquet |
| **License** | CC BY 4.0 |
| **Location** | `data/benchmarks/ragbench/covidqa/` |

The measured test task ranks each row's supplied documents after deduplicating identical
document text. Four test rows have no positive relevance key and are excluded from scored
metrics. Snapshot hash:
`a1dcc76fd4883515c68208e715f4f0488ca4369957e110cbcaf0f52c61134700`.

The validation and test responses are additionally used for a narrow lexical/numeric
source-consistency audit. The validation split selects one threshold; the held-out test
split measures coverage and false certification without retuning. This is not semantic
entailment or truth certification.

---

## Springer ToC (Structural)

| Field | Value |
|---|---|
| **Type** | Publisher Table of Contents (structural) |
| **Source** | Springer BNDR Datasets |
| **Format** | JSON/XML |
| **License** | Commercial — not redistributed |
| **Location** | `data/springer/` |
| **Notes** | Used for GNN extractor demo (no trained weights) |

---

## Fixture Files (Demo/Testing)

Located in `apps/api/tests/fixtures/`:

| File | Purpose |
|---|---|
| `scifact_fixture_cases.json` | 5 labeled SciFact cases (supported/refuted/insufficient) |
| `ragbench_fixture_cases.json` | 3 RAGBench cases |
| `hotpotqa_fixture_cases.json` | 3 multi-hop cases |
| `adversarial_cases.json` | 6 adversarial cases (injection, negation, unit traps, stale source) |

These are hand-crafted fixtures that do NOT require downloading any external data.
They are used by `test_evidence_extraction.py` and `test_text_attribution.py`.

## Provenance boundary

The complete external files are present in the working copy but ignored by Git and
excluded from release archives. Their historical download commands/revisions were not
recorded at acquisition time. The canonical run therefore makes the narrower, auditable
claim that the exact local files were hashed before evaluation, remained unchanged during
evaluation, and had no query-ID overlap across declared splits. It does not claim a
cryptographically verified chain from each upstream server to this checkout.

See [the external benchmark audit](EXTERNAL_BENCHMARK_AUDIT.md) for the complete
classification and validation decisions.

---

## Disclaimer

FaultTrace-RAG is a research platform. Results on benchmarks are produced by
**deterministic rule-based extractors** (not trained LLMs) unless `OPENAI_API_KEY` is set.
Do not compare these results to published state-of-the-art numbers without accounting
for the extraction method used.
