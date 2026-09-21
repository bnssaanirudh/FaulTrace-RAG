# Prompt 3 Completion Report

## 1. Pipeline Definitions (P0-P3)
The research pipeline families P0 through P3 are successfully integrated into `faulttrace_pipelines`:
- **P0-deterministic-scope-baseline**: Uses 100% deterministic scopes and extraction directly mapped to the query predicates and gold evaluation models. No faults.
- **P1-direct-bm25**: Models a standard text-based retrieval fault path. Employs `rank_bm25` (an Okapi BM25 implementation) for context building over the generated dataset, feeding into a single prompt for direct QA.
- **P2-direct-dense**: Extends P1 by substituting BM25 for dense retrieval, employing semantic matching. (Note: Fallbacks to mock embedding generation can be utilized for zero-dependency test execution).
- **P3-extract-aggregate**: The most complex of the research lines. It first retrieves the context, uses the provider LLMs to extract strictly schema-validated JSON, and then uses a deterministic `PandasEvaluator` instance to compute the aggregation metric (`count`, `mean`, etc.) across the extracted nodes.

## 2. Provider Implementations
The Provider interface in `faulttrace_core.llm` now correctly models multiple LLM behaviors:
- **DeterministicProvider**: A predictable, non-network provider designed specifically for `pytest` and CI environments. It always outputs a pre-defined static valid response ensuring local tests finish in milliseconds.
- **OpenAIProvider**: The core production LLM adapter built using the official `openai` Python SDK. It's configured to accept custom base URLs to trivially support **Ollama** instances running locally for privacy-preserving data extraction alongside standard OpenAI deployment.

## 3. Retrieval Technologies & Cache Behavior
- **Retrievers**: `BM25Retriever` handles lexical text search.
- **Cache**: Generated Worlds and queries are permanently persisted locally in Parquet files in the `data/generated/worlds/` directory. For run operations, intermediate context arrays and run traces are cached and logged to the `artifacts/runs` folder, maintaining an immutable config hash and event timeline for debugging or counterfactual attribution.
- Offline gold evaluation is never inadvertently leaked into the prompt construction for P1-P3; we solely use the query's natural language field.

## 4. API & UI Enhancements
- **Providers Router**: Added `/api/v1/providers` to list all registered LLM adapters, and `/api/v1/providers/{provider_id}/test` to dynamically verify their connectivity state prior to pipeline kicks.
- **Pipelines**: The pipelines router natively lists all available schemas (P0-P3).
- **UI & Streaming CLI (Pending/Partial)**: The backend now yields trace events per pipeline run to allow granular stage inspections (latency, tokens consumed, retrieved chunks, extraction status). While the backend produces the full trace tree, integrating the Server-Sent Events (SSE) streaming model into `runs-page.tsx` and tracing UI requires a frontend alignment shift (likely part of Prompt 4 preparation). 

## 5. Testing & Validation
All python backend tests are passing successfully via `pytest apps/api/tests/ -v`.
- The deterministic provider correctly replaces network calls for the pipeline tests, completely avoiding model download or API quota consumption during development.
- The pipeline registry properly registers the newly instantiated 3 research schemas.

## 6. What Remains Mocked
Currently, dense embeddings computation (if unconfigured) defaults to deterministic fast-hashing inside tests. The UI trace panel is currently a snapshot view rather than a live websocket/SSE stream, meaning the UI needs to be refreshed to view updated `create_run` execution statuses for long-running models.

## 7. Migration Path to Prompt 4
Prompt 4 introduces the advanced `CounterfactualAttributor` engine (Shapley Value decomposition). The migration path is highly straightforward because the P0-P3 pipelines already emit discrete, granular trace events with clear boundaries between the Scope, Fact, and Aggregation stages. The Prompt 4 attributor can directly consume the `TraceEvent` models produced by P1-P3 to calculate attribution and locate exactly which phase caused the answer deviation from the dual-gold validator.
