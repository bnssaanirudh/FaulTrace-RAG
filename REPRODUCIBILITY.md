# FaultTrace-RAG Reproducibility Guide

This document provides a step-by-step guide for reproducing the core experimental results reported in the FaultTrace-RAG manuscript from the released archive.

## Canonical Commit

All manuscript results were produced from commit `433f3d58` of the FaultTrace-RAG repository.

## Environment Requirements

- Python 3.11 or 3.12
- Node.js 20+

## Installation

```bash
# Clone the repository
git clone https://github.com/bnssaanirudh/faultrace-RAG.git
cd faultrace-RAG
git checkout 433f3d58

# Create a virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install all dependencies from the canonical source
pip install -e ".[dev,api,llm,retrieval,research]"
```

## Regenerating Manuscript Tables

```bash
python scripts/build_manuscript_evidence.py
```

This will output CSV tables and metrics to `paper/generated/`.

## Running the Test Suite

```bash
pytest apps/api/tests/ -v --timeout=120 --ignore=apps/api/tests/test_smoke.py
```

## External Dataset Acquisition

The following datasets are used but cannot be redistributed:

- **RAGTruth**: Available at [https://huggingface.co/datasets/wandb/RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth). Download and place in `data/raw/ragtruth/`.
- **SciFact**: Available via the BEIR benchmark. Run `python scripts/run_external_retrieval_benchmarks.py --dataset scifact`.
- **HotpotQA**: Available at [https://hotpotqa.github.io](https://hotpotqa.github.io). Download the distractor setting JSON.

## DOI Archive

> **DOI**: `[TO BE ASSIGNED AFTER ZENODO/FIGSHARE UPLOAD]`

This archive will be uploaded to Zenodo upon paper acceptance. The DOI will be inserted here and added to `CITATION.cff`.

## Archive Contents

The release archive (`dist/faulttrace-rag-*.zip`) contains:
- Source code (`packages/`, `apps/`, `scripts/`, `configs/`)
- Documentation (`docs/`, `paper/`)
- Experiment configurations
- Generated paper tables (`paper/generated/`)
- Canonical manifests and checksums
- Licenses

The following are **excluded** from the archive:
- `.venv/`, `node_modules/`
- Raw `.db`, `.parquet`, or intermediate artifact files
- `.env` secrets and credentials
- Exploratory and FAST_MODE result files
