import pytest
from faulttrace_core.models import (
    QuerySpec, PipelineRun, CoverageObservation, EvidenceRequirement,
    CoverageDecision, ReasonCode, AnswerPolicyConfig
)
from faulttrace_pipelines.certification import CertificationEngine

@pytest.fixture
def policy():
    return AnswerPolicyConfig(
        policy_id="strict_exact_v1",
        min_known_scope_coverage=1.0,
        min_extraction_completeness=1.0,
        min_required_field_completeness=1.0
    )

def test_p4_complete_success_certified(policy):
    query = QuerySpec.model_validate({
        "family": "count",
        "natural_language_question": "How many books?",
        "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
        "fact_spec": {"fields": ["rating"]},
        "aggregation_spec": {"kind": "count"},
        "world_id": "test_w"
    })
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=10, raw_answer=10)
    
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
        missing_required_fields=0
    )
    
    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)
    
    assert cert.decision == CoverageDecision.CERTIFIED
    assert ReasonCode.CERTIFIED in cert.reason_codes

def test_p4_missing_extraction_row_abstained(policy):
    query = QuerySpec.model_validate({
        "family": "count",
        "natural_language_question": "How many books?",
        "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
        "fact_spec": {"fields": ["rating"]},
        "aggregation_spec": {"kind": "count"},
        "world_id": "test_w"
    })
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=10, raw_answer=10)
    
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=9, # Missed one row
        missing_required_fields=0
    )
    
    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)
    
    assert cert.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.EXTRACTION_ROWS_MISSING in cert.reason_codes

def test_topk_chance_correct_uncertified(policy):
    query = QuerySpec.model_validate({
        "family": "top_k",
        "natural_language_question": "Top 3 books?",
        "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
        "fact_spec": {"fields": ["title", "rating"]},
        "aggregation_spec": {"kind": "top_k", "k": 3, "group_by_field": "title"},
        "world_id": "test_w"
    })
    run = PipelineRun(query_id=query.query_id, pipeline_id="p1", answer=["A", "B", "C"])
    
    # P1 does not enumerate scope
    obs = CoverageObservation(
        eligible_set_size_known=False,
        retrieved_units=5,
        extracted_valid_rows=5,
        missing_required_fields=0
    )
    
    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)
    
    assert cert.decision == CoverageDecision.UNCERTIFIED
    assert ReasonCode.SCOPE_COVERAGE_UNKNOWN in cert.reason_codes

def test_empty_legitimate_scope(policy):
    query = QuerySpec.model_validate({
        "family": "count",
        "natural_language_question": "How many weird books?",
        "scope_predicate": {"kind": "eq", "field": "category", "value": "WeirdBooks"},
        "fact_spec": {"fields": ["rating"]},
        "aggregation_spec": {"kind": "count"},
        "world_id": "test_w"
    })
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=0, raw_answer=0)
    
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=0,
        unique_represented_record_ids=0,
        retrieved_units=0,
        extracted_valid_rows=0,
        missing_required_fields=0
    )
    
    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)
    
    assert cert.decision == CoverageDecision.CERTIFIED
    assert cert.coverage_ratios["scope_coverage"] == 1.0

def test_policy_version_changes_hash():
    query = QuerySpec.model_validate({
        "family": "count",
        "natural_language_question": "How many books?",
        "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
        "fact_spec": {"fields": ["rating"]},
        "aggregation_spec": {"kind": "count"},
        "world_id": "test_w"
    })
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4")
    obs = CoverageObservation(eligible_set_size_known=True, eligible_set_size=10, unique_represented_record_ids=10, retrieved_units=10, extracted_valid_rows=10)
    
    p1 = AnswerPolicyConfig(policy_id="v1")
    p2 = AnswerPolicyConfig(policy_id="v2")
    
    c1 = CertificationEngine(p1).certify(run, query, obs)
    c2 = CertificationEngine(p2).certify(run, query, obs)
    
    assert c1.certificate_hash != c2.certificate_hash
