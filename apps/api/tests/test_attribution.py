import pytest
import pandas as pd
import json
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

from faulttrace_core.models import (
    PipelineRun, QuerySpec, GoldAnswer, CountSpec, ScopePredicate, FactSpec
)
from faulttrace_gold.oracles import ScopeOracle, ExtractionOracle, AggregationOracle
from faulttrace_pipelines.lattice import OracleLatticeRunner
from faulttrace_pipelines.attribution import CounterfactualAttributor

def _make_dummy_run(answer=10, is_correct=False):
    return PipelineRun(
        run_id=str(uuid4()),
        query_id=str(uuid4()),
        pipeline_id="P4-full-scope-mer",
        provider_id="deterministic",
        status="completed",
        answer=answer,
        gold_answer_value=20,
        is_correct=is_correct,
        is_within_tolerance=is_correct,
        loss=abs(20-answer)/20,
        latency_ms=100.0,
        token_estimate_input=0,
        token_estimate_output=0,
        artifact_references={}
    )

def test_pure_scope_failure():
    # If only R is wrong, then phi_R should be exactly the recoverable error
    # and phi_E = phi_A = 0.
    # Since we can't fully mock the dataframe easily without setting up the artifact_refs 
    # properly in the LatticeRunner, we will just use the evaluator.
    # Actually, we can test exact Shapley math inside the Lattice runner if we monkeypatch `_execute_intervention`.
    
    runner = OracleLatticeRunner()
    
    baseline_loss = 1.0
    def mock_execute(parent_run, query, gold_answer, corpus_df, replace_R, replace_E, replace_A, subset_name):
        # Pure R failure means replacing R fixes everything (if E and A were perfect on their inputs)
        # But wait! If E is perfect on wrong R, does it give the right answer?
        # A pure R failure means:
        # CF(none) = loss 1.0
        # CF(R) = loss 0.0 (because E and A are perfect on the correct R)
        # CF(E) = loss 1.0 (fixing E does nothing if R is still wrong)
        # CF(A) = loss 1.0 (fixing A does nothing)
        # CF(RE) = loss 0.0
        # CF(RA) = loss 0.0
        # CF(EA) = loss 1.0
        # CF(REA) = loss 0.0
        
        from faulttrace_pipelines.lattice import LatticeRun
        from faulttrace_pipelines.loss import LossDiagnostic
        
        if replace_R:
            loss = 0.0
        else:
            loss = 1.0
            
        return LatticeRun(
            intervention_id="test",
            parent_run_id="test_run",
            subset=subset_name,
            answer_value=0,
            loss_diagnostic=LossDiagnostic(normalized_loss=loss, status="valid"),
            status="valid"
        )
        
    runner._execute_intervention = mock_execute
    
    res = runner.execute_lattice(
        parent_run=_make_dummy_run(),
        query=QuerySpec.model_validate({
            "query_id": "q",
            "family": "count",
            "natural_language_question": "How many books?",
            "world_id": "w",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"}
        }),
        gold_answer=GoldAnswer(query_id="q", world_id="w", answer_value=20, tolerance=0.0, record_ids=[], eligible_record_count=10, evidence_hash="dummy"),
        corpus_df=pd.DataFrame()
    )
    
    assert res.phi_R == 1.0
    assert res.phi_E == 0.0
    assert res.phi_A == 0.0
    assert res.interaction == 0.0


def test_leak_guard():
    # Assert that gold concepts are strictly kept to oracles and not in provider config
    from faulttrace_core.llm import ProviderConfig
    config = ProviderConfig(model_id="test", structured_schema={}, structured_schema_name="Test")
    assert not hasattr(config, "gold_answer")
    assert not hasattr(config, "oracle_df")
