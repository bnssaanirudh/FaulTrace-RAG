"""
Coverage adapters to observe and extract evidence metrics from pipeline execution traces.
"""
import pandas as pd
from typing import Optional
from faulttrace_core.models import PipelineRun, CoverageObservation, TraceEvent
from pathlib import Path


def extract_coverage_observations(run: PipelineRun, trace_events: list[TraceEvent], corpus_df: pd.DataFrame) -> CoverageObservation:
    """
    Given a pipeline run and its trace events, extract the evidence coverage metrics.
    """
    obs = CoverageObservation()
    
    # World size
    obs.known_world_size = len(corpus_df)
    
    # Determine eligible set size if known
    # P4/P5 emit scope_enumerate events with the exact eligible count
    scope_event = next((e for e in trace_events if e.event_type == "scope_enumerate"), None)
    if scope_event and scope_event.record_count_out is not None:
        obs.eligible_set_size_known = True
        obs.eligible_set_size = scope_event.record_count_out
        obs.retrieved_units = scope_event.record_count_out
    else:
        # P0 or un-enumerated scopes
        obs.eligible_set_size_known = False
        
    # Extracted rows and uniqueness
    extract_event = next((e for e in trace_events if e.event_type == "fact_extract"), None)
    if extract_event:
        obs.extracted_valid_rows = extract_event.record_count_out or 0
        payload = extract_event.structured_payload
        
        # P4/P5 provide ambiguous/missing via payload
        if "missing_records" in payload:
            obs.failed_rows = len(payload["missing_records"])
            
    # For extraction.parquet, we can directly inspect the artifact if available
    extract_path = run.artifact_references.get("fact_extract")
    if extract_path and Path(extract_path).exists():
        try:
            df_ext = pd.read_parquet(extract_path)
            if "record_id" in df_ext.columns:
                obs.unique_represented_record_ids = df_ext["record_id"].nunique()
            else:
                obs.unique_represented_record_ids = len(df_ext)
                
            # Check missing fields
            # For this simple prototype, if a row exists, we assume fields are present, 
            # or the model dropped them. Let's look for nulls in the parquet.
            if not df_ext.empty:
                null_counts = df_ext.isnull().sum(axis=1)
                obs.missing_required_fields = int((null_counts > 0).sum())
        except Exception:
            pass
    elif extract_event and extract_event.record_count_out:
        obs.unique_represented_record_ids = extract_event.record_count_out
        
    return obs
