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
| **Records** | ~5,000 claims, ~10,000 documents |
| **Format** | JSONL (claims + corpus) |
| **License** | Apache 2.0 |
| **Location** | `data/benchmarks/scifact/` (after download) |
| **Labels** | SUPPORT, REFUTE, NEI (not enough info) |

**To download:**
```bash
cd data/benchmarks
git clone https://github.com/allenai/scifact.git
# Or use the adapter: SciFactAdapter(Path("data/benchmarks")).load_corpus()
```

**Usage notes:**
- `SciFactAdapter` in `packages/data/faulttrace_data/benchmarks/scifact.py` handles loading
- Corpus contains PubMed abstracts; claims are manually annotated
- Use `split="test"` for evaluation; training set is available for future fine-tuning

---

## HotpotQA

| Field | Value |
|---|---|
| **Type** | Multi-hop question answering |
| **Source** | Yang et al., EMNLP 2018 |
| **URL** | https://hotpotqa.github.io/ |
| **Records** | ~113k QA pairs |
| **Format** | JSON |
| **License** | CC BY-SA 4.0 |
| **Location** | `data/benchmarks/hotpotqa/` (after download) |
| **Labels** | Answer string + supporting facts |

**To download:**
```bash
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json -O data/benchmarks/hotpotqa/dev.json
```

**Usage notes:**
- HotpotQA requires multi-hop retrieval (two supporting documents per question)
- The `HotpotQAAdapter` in `packages/data/faulttrace_data/benchmarks/hotpotqa.py` handles loading
- FaultTrace-RAG uses it to test multi-hop evidence chains in the provenance graph

---

## COVID-QA

| Field | Value |
|---|---|
| **Type** | COVID-19 biomedical QA |
| **Source** | CORD-19 / Möller et al., 2020 |
| **URL** | https://huggingface.co/datasets/covid_qa_deepset |
| **Records** | ~2,000 QA pairs from CORD-19 papers |
| **Format** | JSON (SQuAD format) |
| **License** | CC BY 4.0 |
| **Location** | `data/benchmarks/covidqa/` (after download) |

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

---

## Disclaimer

FaultTrace-RAG is a research platform. Results on benchmarks are produced by
**deterministic rule-based extractors** (not trained LLMs) unless `OPENAI_API_KEY` is set.
Do not compare these results to published state-of-the-art numbers without accounting
for the extraction method used.
