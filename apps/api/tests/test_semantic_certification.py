from __future__ import annotations

from pathlib import Path

import pandas as pd
from faulttrace_core.models import (
    AnswerPolicyConfig,
    CoverageDecision,
    PipelineRun,
    QuerySpec,
    ReasonCode,
    TraceEvent,
    TraceEventType,
)
from faulttrace_pipelines.certification import CertificationEngine
from faulttrace_pipelines.coverage_adapters import (
    _measure_structured_fidelity,
    extract_coverage_observations,
)


def _query() -> QuerySpec:
    return QuerySpec.model_validate(
        {
            "family": "mean",
            "natural_language_question": "What is the mean value for books?",
            "scope_predicate": {"kind": "eq", "field": "category", "value": "Books"},
            "fact_spec": {"fields": ["value"]},
            "aggregation_spec": {"kind": "mean", "field": "value"},
            "world_id": "semantic-test-world",
            "tolerance": 1e-6,
        }
    )


def _policy() -> AnswerPolicyConfig:
    return AnswerPolicyConfig(
        policy_id="strict_structured_semantic_v2",
        version="2.0",
        require_provenance_verification=True,
        min_source_fact_fidelity=1.0,
        min_numeric_fidelity=1.0,
        require_aggregation_replay=True,
    )


def _observe(
    tmp_path: Path,
    extraction: pd.DataFrame,
    answer: float,
    corpus: pd.DataFrame,
):
    query = _query()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scope = corpus[corpus["category"] == "Books"].copy()
    scope_path = run_dir / "scope_output.parquet"
    extraction_path = run_dir / "extraction.parquet"
    scope.to_parquet(scope_path, index=False)
    extraction.to_parquet(extraction_path, index=False)
    run = PipelineRun(
        query_id=query.query_id,
        pipeline_id="test",
        answer=answer,
        raw_answer=answer,
        artifact_references={
            "scope_enumerate": str(scope_path),
            "fact_extract": str(extraction_path),
        },
    )
    events = [
        TraceEvent(
            run_id=run.run_id,
            stage="scope_enumerate",
            event_type=TraceEventType.SCOPE_ENUMERATE,
            record_count_out=len(scope),
        ),
        TraceEvent(
            run_id=run.run_id,
            stage="fact_extract",
            event_type=TraceEventType.FACT_EXTRACT,
            record_count_out=len(extraction),
        ),
    ]
    observation = extract_coverage_observations(run, events, corpus, query)
    return CertificationEngine(_policy()).certify(run, query, observation)


def test_semantic_certificate_accepts_source_faithful_replay(tmp_path: Path):
    corpus = pd.DataFrame(
        [
            {"record_id": "r1", "category": "Books", "value": 2.0},
            {"record_id": "r2", "category": "Books", "value": 3.0},
            {"record_id": "r3", "category": "Games", "value": 99.0},
        ]
    )
    extraction = corpus.loc[:1, ["record_id", "value"]].copy()
    certificate = _observe(tmp_path, extraction, 2.5, corpus)

    assert certificate.decision == CoverageDecision.CERTIFIED
    assert certificate.assurance_scope == "structured_semantic"
    assert certificate.coverage_ratios["source_fact_fidelity"] == 1.0
    assert certificate.coverage_ratios["aggregation_replay_consistency"] == 1.0


def test_semantic_certificate_rejects_corrupted_fact(tmp_path: Path):
    corpus = pd.DataFrame(
        [
            {"record_id": "r1", "category": "Books", "value": 2.0},
            {"record_id": "r2", "category": "Books", "value": 3.0},
        ]
    )
    extraction = pd.DataFrame(
        [
            {"record_id": "r1", "value": 20.0},
            {"record_id": "r2", "value": 3.0},
        ]
    )
    certificate = _observe(tmp_path, extraction, 11.5, corpus)

    assert certificate.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.FACT_FIDELITY_BELOW_REQUIRED in certificate.reason_codes
    assert ReasonCode.NUMERIC_FIDELITY_BELOW_REQUIRED in certificate.reason_codes


def test_semantic_certificate_rejects_wrong_aggregation(tmp_path: Path):
    corpus = pd.DataFrame(
        [
            {"record_id": "r1", "category": "Books", "value": 2.0},
            {"record_id": "r2", "category": "Books", "value": 3.0},
        ]
    )
    extraction = corpus[["record_id", "value"]].copy()
    certificate = _observe(tmp_path, extraction, 50.0, corpus)

    assert certificate.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.AGGREGATION_REPLAY_MISMATCH in certificate.reason_codes


def test_semantic_certificate_rejects_invented_provenance(tmp_path: Path):
    corpus = pd.DataFrame(
        [{"record_id": "r1", "category": "Books", "value": 2.0}]
    )
    extraction = pd.DataFrame([{"record_id": "invented", "value": 2.0}])
    certificate = _observe(tmp_path, extraction, 2.0, corpus)

    assert certificate.decision == CoverageDecision.ABSTAIN
    assert ReasonCode.PROVENANCE_MISMATCH in certificate.reason_codes


def test_fact_fidelity_treats_matching_nulls_as_source_faithful():
    query = _query()
    corpus = pd.DataFrame(
        [{"record_id": "r1", "category": "Books", "value": float("nan")}]
    )
    extraction = corpus[["record_id", "value"]].copy()

    provenance, fact_fidelity, numeric_fidelity = _measure_structured_fidelity(
        extraction, corpus, query
    )

    assert provenance == 1.0
    assert fact_fidelity == 1.0
    assert numeric_fidelity is None
