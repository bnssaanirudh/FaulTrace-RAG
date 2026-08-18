import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from faulttrace_core.models import (
    AnswerPolicyConfig,
    CoverageDecision,
    CoverageObservation,
    PipelineRun,
)
from faulttrace_pipelines.certification import CertificationEngine


def mock_extract_semantic_predicates(corpus_text: str, toponyms: str, n: int = 200):
    """
    Mock implementation of Track T semantic extraction (event locations).
    Parses unstructured text and outputs semantic predicates.
    """
    results = []

    # We will simulate high lexical ambiguity on a subset to trigger the coverage certificate
    for i in range(n):
        # Every 5th extraction has high lexical ambiguity
        ambiguous = i % 5 == 0

        results.append(
            {
                "record_id": f"event_track_t_{i}",
                "location_extracted": "Ambiguous City" if ambiguous else f"Resolved City {i}",
                "is_ambiguous": ambiguous,
                "confidence": 0.4 if ambiguous else 0.95,
            }
        )

    return results


def run_track_t():
    print("======================================")
    print("Running Track T Semantic Annotation Pipeline")
    print("======================================")

    data_dir = Path("data")
    corpus_path = data_dir / "event-geoparsing-corpus.txt"
    toponyms_path = data_dir / "toponyms-disambiguated.txt"

    # Read the files if they exist (graceful fallback if they don't for testing)
    corpus_text = (
        corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else "Mock Corpus Text"
    )
    toponyms_text = (
        toponyms_path.read_text(encoding="utf-8") if toponyms_path.exists() else "Mock Toponyms"
    )

    print(f"Loaded Track T Data: {len(corpus_text)} chars from corpus")

    # Execute Extraction
    n_validation = 200
    extracted_data = mock_extract_semantic_predicates(corpus_text, toponyms_text, n=n_validation)

    # Count ambiguous
    ambiguous_count = sum(1 for e in extracted_data if e["is_ambiguous"])
    print(
        f"Extracted {n_validation} predicates, found {ambiguous_count} with high lexical ambiguity."
    )

    # Generate Trace Log & PipelineRun to invoke Coverage Certificate
    run_id = "track_t_val_001"

    run = PipelineRun(
        run_id=run_id,
        query_id="semantic_query_1",
        pipeline_id="track_t_semantic_pipeline",
        provider_id="custom_semantic",
        status="completed",
        answer="Extracted Locations",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_ms=150.0,
    )

    # Create Coverage Observations
    obs = CoverageObservation(
        known_world_size=n_validation,
        eligible_set_size_known=True,
        eligible_set_size=n_validation,
        retrieved_units=n_validation,
        unique_represented_record_ids=n_validation,
        extracted_valid_rows=n_validation - ambiguous_count,
        ambiguous_rows=ambiguous_count,
        failed_rows=0,
    )

    # Initialize Coverage Certificate Module
    # If ambiguity > 10%, it should abstain
    policy = AnswerPolicyConfig(
        policy_id="track_t_strict",
        max_ambiguous_tolerance=0.10,  # Abstain if >10% ambiguous
    )
    cert_engine = CertificationEngine(policy=policy)

    # We pass a dummy query for certification since this is a secondary pipeline
    from faulttrace_core.models import (
        CountSpec,
        EqPredicate,
        FactSpec,
        QuerySpec,
    )

    dummy_query = QuerySpec(
        family="count",
        natural_language_question="Extract all event locations",
        world_id="track_t",
        scope_predicate=EqPredicate(field="dummy", value="dummy"),
        fact_spec=FactSpec(fields=["dummy"]),
        aggregation_spec=CountSpec(kind="count"),
    )

    cert = cert_engine.certify(run, dummy_query, obs)

    print(f"Coverage Certificate Decision: {cert.decision.value}")
    if cert.decision != CoverageDecision.CERTIFIED:
        print(f"Abstention Reasons: {[r.value for r in cert.reason_codes]}")

    run.certificate_id = cert.certificate_id
    run.certificate_hash = cert.certificate_hash
    run.policy_decision = cert.decision.value

    # Save Trace Logs
    out_dir = Path("outputs/track_t")
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = out_dir / f"{run_id}_trace.json"

    output_payload = {
        "run": run.model_dump(mode="json"),
        "certificate": cert.model_dump(mode="json"),
        "observations": obs.model_dump(mode="json"),
    }

    trace_path.write_text(json.dumps(output_payload, indent=2))
    print(f"Saved validation trace log to {trace_path}")


if __name__ == "__main__":
    run_track_t()
