# <h1 align="center">FaultTrace-RAG Analytics Platform</h1>

<p align="center">
  <em>Counterfactual Fault Localization for Corpus-Level LLM Analytics Pipelines.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/node-20+-green.svg" alt="Node Version">
</p>

## 📖 Overview

FaultTrace-RAG addresses a fundamental challenge in LLM-based analytics: when a pipeline produces a wrong answer over a corpus, **which component failed?** Was it the retrieval scope (wrong evidence set R), fact extraction (incorrect E), aggregation logic (wrong A), or an interaction among these?

By systematically replacing each component with a deterministic oracle (using our dual DuckDB/Pandas gold engine), FaultTrace-RAG attributes recoverable errors and issues **Coverage Certificates** for LLM pipeline outputs.

## ✨ Key Features

- **Counterfactual Attribution Engine:** Pinpoints exactly where an LLM failed using Shapley values across Retrieval, Extraction, and Aggregation stages.
- **Dual Gold Evaluation Engine:** Verifies ground-truth query answers against deterministic Pandas and DuckDB evaluations.
- **Selective Prediction Certification:** Issues mathematical coverage certificates indicating whether an LLM's answer is trustable.
- **Track M Synthetic Corpus Generator:** Instantly spins up realistic Amazon-review-style e-commerce worlds to test capabilities at scale.
- **Interactive Next.js Trace Inspector:** Deep dive into LLM context chunks, extraction traces, and tie-breaking policies.
- **Experiment Runner & Hardware Benchmarking:** Run YAML-based specification experiments and automatically benchmark token/latency metrics.
- **Enterprise Security:** Hardened FastAPI backend with CORS, CSP, HSTS, rate-limiting, and path-traversal prevention.

## 🏗️ Architecture

FaultTrace-RAG consists of a distributed architecture centered around decoupling execution from evaluation.

- **`apps/api` (FastAPI):** Python backend serving standard endpoints for worlds, datasets, query generations, and pipeline traces.
- **`apps/web` (Next.js):** React frontend using TailwindCSS and App Router to visualize pipeline metrics and traces.
- **`packages/core`:** Pure Python models, pydantic schemas, and immutable data structures.
- **`packages/gold`:** Dual-engine (DuckDB + Pandas) baseline for deterministic query execution.
- **`packages/reporting`:** Automated academic figure generation (Matplotlib/Seaborn) and metric export.

For a deep dive into the architecture and state model, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/bnssaanirudh/faultrace-RAG.git
cd faultrace-RAG

# Backend Setup
python -m venv .venv
# Windows: .venv\Scripts\activate | Unix: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Frontend Setup
cd apps/web
npm install
cd ../..
```

### 2. Configuration & Initialization

```bash
# Configure environment variables
cp .env.example .env

# Create SQLite database and seed initial demo data
make migrate
make seed-demo
```

*(Note for Windows users: if `make` is unavailable, refer to `INSTALLATION.md` for equivalent PowerShell commands.)*

### 3. Running the Stack

Start the development servers:

```bash
make dev
```
- **API Server:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **Dashboard UI:** http://localhost:3000

## 🧪 Testing and Quality Assurance

FaultTrace-RAG includes a rigorous test suite to guarantee correctness.

```bash
make typecheck   # Mypy and TypeScript strict checks
make lint        # Ruff and ESLint
make test        # Pytest backend suite (Coverage 85%+)
make smoke       # Playwright End-to-End browser tests
```

## 📊 Generating Reports

To benchmark hardware runs and generate academic figures (SVGs and CSVs with watermarks):

```bash
# Generates figures in the `reports/` folder
python scripts/benchmark_runs.py
make generate-figures
```

## 📚 Documentation

For more detailed guides, check out our documentation:
- [User Guide](USER_GUIDE.md): Navigating the UI and running experiments.
- [Installation Guide](INSTALLATION.md): Detailed OS-specific installation instructions.
- [Decisions](docs/DECISIONS.md): Architectural Decision Records (ADRs).
- [Build State](docs/BUILD_STATE.md): Component-level implementation status.

## 🤝 Contributing

Contributions are welcome! Please ensure you run `make release-check` to verify your changes against our quality gates before submitting a Pull Request.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🎓 Citation

If you use FaultTrace-RAG in your research, please cite it using the provided `CITATION.cff` file or the BibTeX format below:

```bibtex
@software{FaultTrace_RAG_2026,
  author = {FaultTrace-RAG Contributors},
  title = {FaultTrace-RAG: Counterfactual Fault Localization for LLM Pipelines},
  year = {2026},
  url = {https://github.com/bnssaanirudh/faultrace-RAG}
}
```
