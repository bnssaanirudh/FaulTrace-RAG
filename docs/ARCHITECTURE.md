# Architecture: FaultTrace-RAG

## 1. System Purpose
FaultTrace-RAG is a principled counterfactual diagnosis and repair framework for multi-stage analytical Retrieval-Augmented Generation (RAG) pipelines. It replaces unprincipled manual prompt-engineering by algorithmically isolating the exact pipeline component (Retrieval, Extraction, Aggregation) responsible for a downstream hallucination or failure. Using exact Shapley values and descendant replay, the framework allows operators to quantify blame, repair localized faults within bounded budgets (BACD/MCR), and certify system-level outcomes.

## 2. Logical Layers
The system is cleanly decoupled into the following logical strata:
- **Data**: Handlers for synthetic metadata and external domain corpora (e.g., SciFact, RAGTruth, COVID-QA).
- **Query Specification**: AST-like representation of analytical requirements.
- **Gold Engine**: Deterministic query runners that execute against pristine datasets to generate ground-truth `GoldAnswer` objects.
- **Pipeline Runtime**: The orchestrator that executes the `Pipeline` interface and manages step-by-step state and lineage.
- **Retrieval**: Modules for direct context, BM25, and dense embedding retrieval.
- **Extraction**: Extractors bridging unstructured text to structured facts via constrained generation.
- **Aggregation/Generation**: Synthesizers that reduce extraction sets into final answers.
- **Counterfactual Engine**: Oracle-replacement operators that inject ground truth at targeted stages to measure downstream answer loss recovery.
- **Active Diagnosis**: The Bayesian/Greedy policies that optimally select counterfactual probes to minimize uncertainty.
- **Repair**: The Cost-Aware and Minimum Counterfactual Repair engines that determine the cheapest set of component updates to flip a bad answer.
- **Certification**: The `CoverageCertificate` layer that performs strict source-and-program consistency checks.
- **Reporting**: Parsers and bootstrapping scripts (Python/R) that render experiment artifacts into tables and plots.
- **API**: A FastAPI service exposing endpoints for real-time trace, diagnosis, and repair workflows.
- **Frontend**: A Next.js application providing interactive trace trees, counterfactual heatmaps, and annotation queues.
- **Artifact Storage**: Persistent, immutable trace records utilizing SQLite (relational) and Parquet (analytical sweeps).

## 3. Current Pipeline Graph
The framework treats RAG as a directed acyclic graph of functional components:
- **Core 3-Stage**: `R → E → A` (Retrieval → Extraction → Aggregation)
- **Generalized \(C_1 \dots C_n\)**: The runtime supports arbitrary depth via a list of `PipelineStage` objects.
- **Experimental 5-Stage**: `S → R → E → A → G` (Scope → Retrieval → Extraction → Aggregation → Generation), enabling deep sub-component targeting (e.g., distinguishing semantic generation `G` from factual aggregation `A`).

## 4. Counterfactual Execution
The core diagnostic mechanism is **descendant replay**. To evaluate the fault at stage $i$, the engine dynamically injects the oracle output $O_i^*$ at stage $i$, and re-executes all downstream components $j > i$. Caching ensures components $k < i$ are not redundantly re-run. This isolation isolates the exact downstream consequence of upstream faults, eliminating observational confounding.

## 5. Artifact Lineage
All pipeline executions are completely reproducible.
- **Run Artifacts**: Outputs, inputs, logs, and token usage for each component are recorded.
- **Hashes**: Configuration and trace states are secured via cryptographic SHA-256 hashes.
- **Manifests**: Large sweeps output deterministic Parquet files, tracked via `MANIFEST_SHA256.json`.
- **Immutable Bundles**: Once an experiment completes, its artifact directory is sealed; analysis scripts compute properties strictly offline.

## 6. Certification
Certificates in FaultTrace-RAG are strict structural constraints, not universal truth guarantees.
- **Certified**: The final answer is perfectly grounded in the provided evidence scope, and the aggregation conforms to the source format without hallucinations.
- **Uncertified**: The framework detected lexical/numeric drift, hallucinated entities, or an unknown semantic envelope.
- **What is NOT certified**: Real-world scientific truth, dataset-level fairness, or open-ended generative quality. Negative statistical risk sweeps confirm that these are empirical operating points bounded by the experimental environment.

## 7. Experiment Architecture
To support the manuscript and external validation, experiments are partitioned:
- **Controlled Synthetic**: Track-M experiments utilizing the internal generator for perfect exact-match evaluations.
- **External Retrieval**: SciFact, HotpotQA, and COVID-QA pipelines leveraging external datasets.
- **Deterministic Mechanics**: Tests isolating infrastructure integrity without relying on non-deterministic LLMs.
- **PAPER_MODE**: The strict canonical configuration used to produce the 488,250-case benchmark result tables.
- **Live-LLM**: 200 audited natural-failure cases testing BACD/MCR against live Qwen/Mistral inferences.
- **RAGTruth**: Statistical sweeps explicitly searching for risk-controlled false-certification bounds.

## 8. Service Architecture
The full-stack application relies on:
- **FastAPI** (Python 3.12) as the backend orchestrator.
- **Next.js** (React) as the dashboard and attribution visualization frontend.
- **SQLite / Parquet**: SQLite for rapid trace queries via the UI; Parquet for large-scale R DataFrame consumption.

## 9. Security Boundary
The application is **Local-only by default**. It runs without auth mechanisms, intending to be a developer tool on a secure researcher workstation. The system has access to execute untrusted text streams via LLM prompts. Running the frontend or API bound to public interfaces without a reverse proxy or auth barrier is strictly discouraged.

## 10. Extension Points
The architecture is designed to be highly extensible via subclassing:
- **New Stage**: Inherit from `PipelineStage` and implement `execute()`.
- **New Retriever**: Subclass `BaseRetriever` and implement `retrieve(query)`.
- **New Oracle**: Subclass `GoldEngine` and implement a deterministic evaluator for your data schema.
- **New Provider**: Subclass `ModelProvider` to support new external LLM endpoints (e.g., Azure, Bedrock, vLLM).
- **New Dataset**: Place raw JSONL/CSV files in `data/raw/` and implement a data loader script mirroring `load_scifact()`.
