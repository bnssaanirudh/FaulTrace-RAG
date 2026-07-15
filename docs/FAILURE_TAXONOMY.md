# Failure Taxonomy

This glossary defines the stable failure codes utilized in FaultTrace-RAG to track and categorize errors during pipeline execution.

## Scope Extraction Failures
- `scope_compilation_failure`: The AST predicate could not be compiled into a deterministic execution filter for the data engine (e.g. Pandas or DuckDB).
- `unsupported_semantic_predicate`: A semantic Track T predicate was encountered which currently lacks structured metadata resolution capability.

## Model / Environment Failures
- `rendering_failure`: The template engine failed to render the context or prompt for a given extraction unit.
- `provider_timeout`: The LLM inference provider (OpenAI, Ollama, etc.) timed out or rate-limited the execution beyond configured retries.

## Structured Extraction Failures
- `invalid_structured_output`: The model failed to output syntactically valid JSON, or the output violated the expected structured schema.
- `invented_record_id`: The model returned a JSON record mapped to a `record_id` that was not present in the original chunk sent in the prompt.
- `missing_record_id`: The model silently omitted a `record_id` that was expected to be extracted based on the scoped input batch.
- `field_validation_failure`: One or more extracted fields failed strict type validation (e.g., returning a string instead of a float).

## Aggregation & Reduction Failures
- `unresolved_ambiguity`: The model explicitly marked a record as ambiguous (`scope_decision = "ambiguous"`), and bounded repair (P5) could not resolve it.
- `incomplete_map_coverage`: P4/P5 finished execution, but the output coverage did not meet 100% of the enumerated input scope requirement.
- `deterministic_reducer_failure`: The `PandasEvaluator` failed to apply the `AggregationSpec` over the extracted JSON rows.
- `gold_comparison_unavailable`: The pipeline could not evaluate its extracted answer against the ground truth because the gold artifact is missing or un-evaluable.

## Data & System Failures
- `artifact_integrity_failure`: Hashes mismatched between cache, schema, or runtime artifacts, indicating corrupted storage or cross-version collisions.
