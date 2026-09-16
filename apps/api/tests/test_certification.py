import pytest
from faulttrace_core.models import (
    AnswerPolicyConfig,
    CoverageDecision,
    CoverageObservation,
    PipelineRun,
    QuerySpec,
    ReasonCode,
)
from faulttrace_pipelines.certification import CertificationEngine


@pytest.fixture
def policy():
    return AnswerPolicyConfig(
        policy_id="strict_exact_v1",
        min_known_scope_coverage=1.0,
        min_extraction_completeness=1.0,
        min_required_field_completeness=1.0,
    )


def test_p4_complete_success_certified(policy):
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=10, raw_answer=10)

    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
        missing_required_fields=0,
    )

    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)

    assert cert.decision == CoverageDecision.CERTIFIED
    assert ReasonCode.CERTIFIED in cert.reason_codes


def test_p4_missing_extraction_row_abstained(policy):
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=10, raw_answer=10)

    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=9,  # Missed one row
        missing_required_fields=0,
    )

    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)

    assert cert.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.EXTRACTION_ROWS_MISSING in cert.reason_codes


def test_topk_chance_correct_uncertified(policy):
    query = QuerySpec.model_validate(
        {
            "family": "top_k",
            "natural_language_question": "Top 3 books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["title", "rating"]},
            "aggregation_spec": {"kind": "top_k", "k": 3, "group_by_field": "title"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p1", answer=["A", "B", "C"])

    # P1 does not enumerate scope
    obs = CoverageObservation(
        eligible_set_size_known=False,
        retrieved_units=5,
        extracted_valid_rows=5,
        missing_required_fields=0,
    )

    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)

    assert cert.decision == CoverageDecision.UNCERTIFIED
    assert ReasonCode.SCOPE_COVERAGE_UNKNOWN in cert.reason_codes


def test_empty_legitimate_scope(policy):
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many weird books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "WeirdBooks"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=0, raw_answer=0)

    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=0,
        scope_membership_known=True,
        unique_represented_record_ids=0,
        retrieved_units=0,
        extracted_valid_rows=0,
        missing_required_fields=0,
    )

    engine = CertificationEngine(policy)
    cert = engine.certify(run, query, obs)

    assert cert.decision == CoverageDecision.CERTIFIED
    assert cert.coverage_ratios["scope_coverage"] == 1.0


def test_policy_version_changes_hash():
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4")
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
    )

    p1 = AnswerPolicyConfig(policy_id="v1")
    p2 = AnswerPolicyConfig(policy_id="v2")

    c1 = CertificationEngine(p1).certify(run, query, obs)
    c2 = CertificationEngine(p2).certify(run, query, obs)

    assert c1.certificate_hash != c2.certificate_hash


def test_ambiguity_threshold_is_enforced(policy):
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many books are unambiguous?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=9)
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
        ambiguous_rows=1,
    )
    cert = CertificationEngine(policy).certify(run, query, obs)
    assert cert.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.EXTRACTION_AMBIGUOUS in cert.reason_codes


def test_topk_requires_candidate_and_tie_completeness(policy):
    query = QuerySpec.model_validate(
        {
            "family": "top_k",
            "natural_language_question": "Which are the top three books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["title"]},
            "aggregation_spec": {"kind": "top_k", "k": 3, "group_by_field": "title"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p4", answer=["A", "B", "C"])
    obs = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=10,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
        ranking_candidate_completeness=0.8,
        tie_boundary_completeness=False,
    )
    cert = CertificationEngine(policy).certify(run, query, obs)
    assert ReasonCode.RANKING_DOMAIN_INCOMPLETE in cert.reason_codes
    assert ReasonCode.TIE_BOUNDARY_UNRESOLVED in cert.reason_codes


def test_wrong_scope_membership_abstains_even_when_counts_match(policy):
    query = QuerySpec.model_validate(
        {
            "family": "count",
            "natural_language_question": "How many books are present?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["rating"]},
            "aggregation_spec": {"kind": "count"},
            "world_id": "test_w",
        }
    )
    run = PipelineRun(query_id=query.query_id, pipeline_id="p1", answer=10)
    observation = CoverageObservation(
        eligible_set_size_known=True,
        eligible_set_size=10,
        scope_membership_known=True,
        eligible_record_ids_covered=8,
        unexpected_record_ids=2,
        unique_represented_record_ids=10,
        retrieved_units=10,
        extracted_valid_rows=10,
    )
    certificate = CertificationEngine(policy).certify(run, query, observation)
    assert certificate.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.SCOPE_COVERAGE_BELOW_REQUIRED in certificate.reason_codes
    assert certificate.coverage_ratios["scope_coverage"] == 0.8
    assert certificate.coverage_ratios["scope_precision"] == 0.8
