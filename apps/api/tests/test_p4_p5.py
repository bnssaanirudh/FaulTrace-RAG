import pytest
import json
import pandas as pd
from pathlib import Path
from faulttrace_core.models import QuerySpec, GoldAnswer
from faulttrace_pipelines.p4_full_scope_mer import P4FullScopeMERPipeline
from faulttrace_pipelines.p5_certified_mer import P5CertifiedMERPipeline
import faulttrace_pipelines.llm.deterministic  # Register provider

# Dummy query and df for testing
@pytest.fixture
def dummy_query():
    spec_json = {
        "family": "mean",
        "natural_language_question": "What is the average rating?",
        "scope_predicate": {
            "kind": "eq",
            "field": "category",
            "value": "Electronics"
        },
        "fact_spec": {
            "fields": ["rating"],
            "derived_fields": [],
            "null_policy": "exclude"
        },
        "aggregation_spec": {
            "kind": "mean",
            "field": "rating"
        },
        "world_id": "world-test",
        "template_id": "manual"
    }
    return QuerySpec.model_validate(spec_json)

@pytest.fixture
def dummy_df():
    data = [
        {"record_id": "r1", "category": "Electronics", "rating": 4.0},
        {"record_id": "r2", "category": "Books", "rating": 5.0},
        {"record_id": "r3", "category": "Electronics", "rating": 2.0},
    ]
    return pd.DataFrame(data)

def test_p4_full_scope_mer(dummy_query, dummy_df, tmp_path):
    pipeline = P4FullScopeMERPipeline(artifacts_dir=tmp_path)
    ans, events, components, tin, tout = pipeline._execute("test-run", dummy_query, dummy_df, None, None)
    
    # Check that events exist
    assert any(e.stage == "scope_enumerate" for e in events)
    assert any(e.stage == "fact_extract" for e in events)
    assert any(e.stage == "aggregate" for e in events)
    
def test_p5_certified_mer(dummy_query, dummy_df, tmp_path):
    pipeline = P5CertifiedMERPipeline(artifacts_dir=tmp_path)
    ans, events, components, tin, tout = pipeline._execute("test-run-p5", dummy_query, dummy_df, None, None)
    
    # Check events
    assert any(e.stage == "fact_extract" for e in events)
    
