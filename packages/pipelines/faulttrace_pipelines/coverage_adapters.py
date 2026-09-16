"""
Coverage adapters to observe and extract evidence metrics from pipeline execution traces.
"""

from pathlib import Path

import pandas as pd
from faulttrace_core.models import (
    CoverageObservation,
    PipelineRun,
    ProportionSpec,
    QuerySpec,
    TopKSpec,
    TraceEvent,
    TrendSpec,
)
from faulttrace_core.predicates import compiler
from faulttrace_gold.pandas_engine import PandasEvaluator
from faulttrace_gold.validator import results_agree


def _is_numeric(value: object) -> bool:
    """Return true for finite scalar numbers, excluding booleans."""
    if isinstance(value, bool):
        return False
    try:
        return bool(pd.notna(value)) and isinstance(value, int | float)
    except (TypeError, ValueError):
        return False


def _measure_structured_fidelity(
    extracted_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    query: QuerySpec,
) -> tuple[float | None, float | None, float | None]:
    """Measure provenance, fact, and numeric fidelity without consulting gold answers.

    Values in the extraction artifact are compared with their immutable source
    records using ``record_id``. The query tolerance is used recursively for
    numeric cells. Missing/duplicate provenance is conservatively counted as a
    mismatch rather than silently discarded.
    """
    if "record_id" not in extracted_df.columns or "record_id" not in corpus_df.columns:
        return None, None, None
    if extracted_df.empty:
        return 1.0, 1.0, 1.0

    source = corpus_df.copy()
    source["record_id"] = source["record_id"].astype(str)
    extracted = extracted_df.copy()
    extracted["record_id"] = extracted["record_id"].astype(str)
    unique_source = source.drop_duplicates("record_id", keep=False).set_index("record_id")

    valid_ids = extracted["record_id"].isin(unique_source.index)
    provenance_coverage = float(valid_ids.mean())
    required_fields = [field for field in query.fact_spec.fields if field in extracted.columns]
    if not query.fact_spec.fields:
        return provenance_coverage, 1.0, None
    comparable_fields = [field for field in required_fields if field in unique_source.columns]
    if not comparable_fields:
        return provenance_coverage, None, None

    matched = 0
    total = len(extracted) * len(comparable_fields)
    numeric_matched = 0
    numeric_total = 0
    for row in extracted.itertuples(index=False):
        record_id = str(row.record_id)
        if record_id not in unique_source.index:
            continue
        source_row = unique_source.loc[record_id]
        for field in comparable_fields:
            extracted_value = getattr(row, field)
            source_value = source_row[field]
            both_missing = (
                pd.api.types.is_scalar(extracted_value)
                and pd.api.types.is_scalar(source_value)
                and bool(pd.isna(extracted_value))
                and bool(pd.isna(source_value))
            )
            agrees = both_missing or results_agree(
                extracted_value, source_value, query.tolerance
            )
            matched += int(agrees)
            if _is_numeric(extracted_value) or _is_numeric(source_value):
                numeric_total += 1
                numeric_matched += int(agrees)

    fact_fidelity = matched / total if total else None
    numeric_fidelity = numeric_matched / numeric_total if numeric_total else None
    return provenance_coverage, fact_fidelity, numeric_fidelity


def _measure_aggregation_replay(
    extracted_df: pd.DataFrame,
    run: PipelineRun,
    query: QuerySpec,
) -> bool | None:
    """Replay the declared aggregation from extracted rows and compare to output."""
    try:
        replay_df = extracted_df
        if "scope_decision" in replay_df.columns:
            replay_df = replay_df[replay_df["scope_decision"] == "in_scope"]
        replayed, _, _ = PandasEvaluator()._aggregate(
            query.aggregation_spec, replay_df, query
        )
        return results_agree(replayed, run.answer, query.tolerance)
    except Exception:
        return None


def extract_coverage_observations(
    run: PipelineRun,
    trace_events: list[TraceEvent],
    corpus_df: pd.DataFrame,
    query: QuerySpec | None = None,
) -> CoverageObservation:
    """
    Given a pipeline run and its trace events, extract the evidence coverage metrics.
    """
    obs = CoverageObservation()

    # World size
    obs.known_world_size = len(corpus_df)

    # Determine the expected eligible set from the executable query whenever
    # possible. A pipeline's own row-count report cannot certify its scope.
    scope_event = next((e for e in trace_events if e.event_type == "scope_enumerate"), None)
    if query is not None and "record_id" in corpus_df.columns:
        try:
            expected_mask = compiler.to_pandas_mask(query.scope_predicate, corpus_df)
            expected_ids = set(corpus_df.loc[expected_mask, "record_id"].astype(str))
            obs.eligible_set_size_known = True
            obs.eligible_set_size = len(expected_ids)
        except Exception:
            expected_ids = None
    else:
        expected_ids = None

    if scope_event and scope_event.record_count_out is not None:
        obs.retrieved_units = scope_event.record_count_out

    scope_path = run.artifact_references.get("scope_enumerate") or run.artifact_references.get(
        "scope_output"
    )
    if scope_path and Path(scope_path).exists() and expected_ids is not None:
        try:
            scope_df = pd.read_parquet(scope_path, columns=["record_id"])
            retrieved_ids = set(scope_df["record_id"].astype(str))
            obs.scope_membership_known = True
            obs.retrieved_units = len(scope_df)
            obs.eligible_record_ids_covered = len(retrieved_ids & expected_ids)
            obs.unexpected_record_ids = len(retrieved_ids - expected_ids)
        except Exception:
            pass

    # Extracted rows and uniqueness
    extract_event = next((e for e in trace_events if e.event_type == "fact_extract"), None)
    if extract_event:
        obs.extracted_valid_rows = extract_event.record_count_out or 0
        payload = extract_event.structured_payload

        # P4/P5 provide ambiguous/missing via payload
        if "missing_records" in payload:
            obs.failed_rows = len(payload["missing_records"])
        obs.ambiguous_rows = int(payload.get("ambiguous_rows", 0) or 0)
        obs.truncation_count = int(payload.get("truncation_count", 0) or 0)
        obs.dropped_context_count = int(payload.get("dropped_context_count", 0) or 0)

    # For extraction.parquet, we can directly inspect the artifact if available
    extract_path = run.artifact_references.get("fact_extract") or run.artifact_references.get(
        "extraction"
    )
    if extract_path and Path(extract_path).exists():
        try:
            df_ext = pd.read_parquet(extract_path)
            if "record_id" in df_ext.columns:
                obs.unique_represented_record_ids = df_ext["record_id"].nunique()
            else:
                obs.unique_represented_record_ids = len(df_ext)

            if "scope_decision" in df_ext.columns:
                obs.ambiguous_rows = int((df_ext["scope_decision"] == "ambiguous").sum())
                obs.failed_rows = max(
                    obs.failed_rows,
                    int((df_ext["scope_decision"] == "missing_evidence").sum()),
                )
                obs.extracted_valid_rows = int((df_ext["scope_decision"] == "in_scope").sum())

            # Only required fact fields participate in completeness.
            if not df_ext.empty:
                required_fields = query.fact_spec.fields if query is not None else list(df_ext.columns)
                present_required = [field for field in required_fields if field in df_ext.columns]
                missing_columns = len(set(required_fields) - set(present_required)) * len(df_ext)
                null_cells = int(df_ext[present_required].isnull().sum().sum()) if present_required else 0
                obs.missing_required_fields = missing_columns + null_cells

            if query is not None:
                provenance, fact_fidelity, numeric_fidelity = _measure_structured_fidelity(
                    df_ext, corpus_df, query
                )
                obs.provenance_verifiable = provenance is not None
                obs.provenance_coverage = provenance
                obs.source_fact_fidelity = fact_fidelity
                obs.numeric_fidelity = numeric_fidelity
                replay_consistent = _measure_aggregation_replay(df_ext, run, query)
                obs.aggregation_replay_evaluable = replay_consistent is not None
                obs.aggregation_replay_consistent = replay_consistent
        except Exception:
            pass
    elif extract_event and extract_event.record_count_out:
        obs.unique_represented_record_ids = extract_event.record_count_out

    full_scope = (
        obs.scope_membership_known
        and obs.eligible_set_size_known
        and obs.eligible_set_size is not None
        and obs.eligible_record_ids_covered == obs.eligible_set_size
        and obs.unexpected_record_ids == 0
    )
    complete_extraction = obs.retrieved_units == obs.extracted_valid_rows and obs.failed_rows == 0
    derivation_complete = full_scope and complete_extraction and run.answer is not None

    if query is not None and isinstance(query.aggregation_spec, ProportionSpec):
        obs.denominator_evaluable = derivation_complete
        obs.numerator_evaluable = derivation_complete
    if query is not None and isinstance(query.aggregation_spec, TopKSpec):
        obs.ranking_candidate_completeness = 1.0 if derivation_complete else 0.0
        obs.tie_boundary_completeness = derivation_complete
    if query is not None and isinstance(query.aggregation_spec, TrendSpec):
        obs.time_bucket_completeness = 1.0 if derivation_complete else 0.0

    return obs
