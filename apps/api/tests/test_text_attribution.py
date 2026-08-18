"""
Tests for TextAttributor and BenchmarkErrorTaxonomy.

Tests:
- Known injected R/E/A failures map to expected dominant_fault
- Negative Shapley values are preserved (NOT clamped to 0)
- Attribution math: phi_R + phi_E + phi_A + interaction == total_recoverable_error
- Failure taxonomy classification (single vs compound faults)
- support_status_loss maps correctly
- P0-P5 regression: existing AttributionResult fields still present (backward compat)
"""

from faulttrace_pipelines.text_attribution import (
    MixedError,
    TextAttributor,
    support_status_loss,
)

# ---------------------------------------------------------------------------
# support_status_loss mapping
# ---------------------------------------------------------------------------


class TestSupportStatusLoss:
    def test_supported_zero_loss(self):
        assert support_status_loss("supported") == 0.0

    def test_partially_supported_half_loss(self):
        assert support_status_loss("partially_supported") == 0.5

    def test_insufficient_evidence_half_loss(self):
        assert support_status_loss("insufficient_evidence") == 0.5

    def test_conflicting_high_loss(self):
        assert support_status_loss("conflicting") == 0.8

    def test_unsupported_full_loss(self):
        assert support_status_loss("unsupported") == 1.0

    def test_unknown_status_defaults_to_half(self):
        assert support_status_loss("unknown_status_xyz") == 0.5


# ---------------------------------------------------------------------------
# TextAttributor — correct answer (baseline loss = 0)
# ---------------------------------------------------------------------------


class TestTextAttributorCorrectAnswer:
    def test_correct_answer_zero_attribution(self):
        """When pipeline_support_status == 'supported', all phi = 0."""
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q1",
            dataset_id="scifact",
            pipeline_id="text-answer",
            pipeline_answer="Drug X reduces risk",
            pipeline_support_status="supported",
            gold_support_status="supported",
            oracle_results={
                "none": "supported",
                "R": "supported",
                "E": "supported",
                "A": "supported",
                "RE": "supported",
                "RA": "supported",
                "EA": "supported",
                "REA": "supported",
            },
        )
        assert result.phi_retrieval == 0.0
        assert result.phi_extraction == 0.0
        assert result.phi_aggregation == 0.0
        assert result.interaction_term == 0.0
        assert result.dominant_fault == "none"


# ---------------------------------------------------------------------------
# TextAttributor — known retrieval failure
# ---------------------------------------------------------------------------


class TestTextAttributorRetrievalFault:
    def test_retrieval_failure_dominant(self):
        """Oracle R fixes the error → R has high Shapley value."""
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q2",
            dataset_id="scifact",
            pipeline_id="text-answer",
            pipeline_answer=None,
            pipeline_support_status="insufficient_evidence",  # loss = 0.5
            gold_support_status="supported",
            oracle_results={
                "none": "insufficient_evidence",  # baseline: 0.5
                "R": "supported",  # R oracle fixes it: 0.0
                "E": "insufficient_evidence",  # E alone doesn't help
                "A": "insufficient_evidence",  # A alone doesn't help
                "RE": "supported",
                "RA": "supported",
                "EA": "insufficient_evidence",
                "REA": "supported",
            },
        )
        assert result.phi_retrieval > result.phi_extraction
        assert result.phi_retrieval > result.phi_aggregation
        assert result.dominant_fault == "retrieval"

    def test_retrieval_fault_failure_category(self):
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q3",
            dataset_id="scifact",
            pipeline_id="text-answer",
            pipeline_answer=None,
            pipeline_support_status="insufficient_evidence",
            gold_support_status="supported",
            oracle_results={
                "none": "insufficient_evidence",
                "R": "supported",
                "E": "insufficient_evidence",
                "A": "insufficient_evidence",
                "RE": "supported",
                "RA": "supported",
                "EA": "insufficient_evidence",
                "REA": "supported",
            },
        )
        assert result.failure_category == "retrieval"


# ---------------------------------------------------------------------------
# TextAttributor — known extraction failure
# ---------------------------------------------------------------------------


class TestTextAttributorExtractionFault:
    def test_extraction_failure_dominant(self):
        """Oracle E fixes error → E has highest Shapley value."""
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q4",
            dataset_id="scifact",
            pipeline_id="text-extract",
            pipeline_answer=None,
            pipeline_support_status="unsupported",  # loss = 1.0
            gold_support_status="supported",
            oracle_results={
                "none": "unsupported",
                "R": "unsupported",  # R alone doesn't help
                "E": "supported",  # E oracle fixes it
                "A": "unsupported",
                "RE": "supported",
                "RA": "unsupported",
                "EA": "supported",
                "REA": "supported",
            },
        )
        assert result.phi_extraction > result.phi_retrieval
        assert result.phi_extraction > result.phi_aggregation
        assert result.dominant_fault == "extraction"


# ---------------------------------------------------------------------------
# Negative Shapley values are NOT clamped
# ---------------------------------------------------------------------------


class TestNegativeShapleyValues:
    def test_negative_phi_not_clamped(self):
        """
        If oracle R makes the answer WORSE, phi_R should be negative.
        This tests that we removed the max(0.0, ...) clamp.
        """
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q5",
            dataset_id="scifact",
            pipeline_id="text-answer",
            pipeline_answer="Drug X works",
            pipeline_support_status="partially_supported",  # baseline loss = 0.5
            gold_support_status="supported",
            oracle_results={
                "none": "partially_supported",
                "R": "unsupported",  # Oracle R makes things WORSE (loss goes from 0.5 → 1.0)
                "E": "supported",  # Oracle E fixes
                "A": "partially_supported",
                "RE": "partially_supported",
                "RA": "unsupported",
                "EA": "supported",
                "REA": "supported",
            },
        )
        # phi_R should be negative (oracle R worsened the answer)
        assert result.phi_retrieval < 0.0, (
            f"Expected negative phi_R but got {result.phi_retrieval}. "
            "This suggests Shapley values are still being clamped to 0."
        )
        assert "retrieval" in result.negative_contributions

    def test_negative_phi_in_negative_contributions_list(self):
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q6",
            dataset_id="test",
            pipeline_id="text-answer",
            pipeline_answer=None,
            pipeline_support_status="partially_supported",
            gold_support_status="supported",
            oracle_results={
                "none": "partially_supported",
                "R": "unsupported",  # harmful oracle
                "E": "supported",
                "A": "partially_supported",
                "RE": "partially_supported",
                "RA": "unsupported",
                "EA": "supported",
                "REA": "supported",
            },
        )
        assert len(result.negative_contributions) > 0
        assert "retrieval" in result.negative_contributions


# ---------------------------------------------------------------------------
# Attribution Efficiency Axiom
# ---------------------------------------------------------------------------


class TestAttributionEfficiencyAxiom:
    def test_phi_sum_plus_interaction_equals_recoverable(self):
        """
        phi_R + phi_E + phi_A + interaction == total_recoverable_error
        This is the Shapley efficiency axiom.
        """
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q7",
            dataset_id="scifact",
            pipeline_id="text-extract",
            pipeline_answer=None,
            pipeline_support_status="conflicting",  # loss = 0.8
            gold_support_status="supported",
            oracle_results={
                "none": "conflicting",
                "R": "partially_supported",
                "E": "partially_supported",
                "A": "supported",
                "RE": "supported",
                "RA": "supported",
                "EA": "supported",
                "REA": "supported",
            },
        )
        total_phi = result.phi_retrieval + result.phi_extraction + result.phi_aggregation
        reconstructed = total_phi + result.interaction_term
        assert abs(reconstructed - result.total_recoverable_error) < 1e-9, (
            f"Efficiency axiom violated: "
            f"phi_R+phi_E+phi_A+interaction={reconstructed:.9f} "
            f"!= total_recoverable_error={result.total_recoverable_error:.9f}"
        )


# ---------------------------------------------------------------------------
# Failure taxonomy classification
# ---------------------------------------------------------------------------


class TestFailureTaxonomy:
    def test_single_retrieval_fault_classified(self):
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q8",
            dataset_id="test",
            pipeline_id="text",
            pipeline_answer=None,
            pipeline_support_status="insufficient_evidence",
            gold_support_status="supported",
            oracle_results={
                "none": "insufficient_evidence",
                "R": "supported",
                "E": "insufficient_evidence",
                "A": "insufficient_evidence",
                "RE": "supported",
                "RA": "supported",
                "EA": "insufficient_evidence",
                "REA": "supported",
            },
        )
        assert result.failure_category == "retrieval"

    def test_compound_r_e_fault_classified(self):
        """Mixed R+E failure produces compound failure category."""
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q9",
            dataset_id="test",
            pipeline_id="text",
            pipeline_answer=None,
            pipeline_support_status="unsupported",  # loss = 1.0
            gold_support_status="supported",
            oracle_results={
                "none": "unsupported",
                "R": "partially_supported",  # R helps some
                "E": "partially_supported",  # E helps some
                "A": "unsupported",  # A alone doesn't help
                "RE": "supported",  # R+E together fix it
                "RA": "partially_supported",
                "EA": "partially_supported",
                "REA": "supported",
            },
        )
        assert result.failure_category in (MixedError.R_PLUS_E.value, "retrieval", "extraction")

    def test_no_fault_when_correct(self):
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q10",
            dataset_id="test",
            pipeline_id="text",
            pipeline_answer="correct",
            pipeline_support_status="supported",
            gold_support_status="supported",
            oracle_results=dict.fromkeys(
                ["none", "R", "E", "A", "RE", "RA", "EA", "REA"], "supported"
            ),
        )
        assert result.dominant_fault == "none"
        assert result.failure_category == "none"


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


class TestTextAttributionResultSerialization:
    def test_to_dict_contains_all_required_fields(self):
        attributor = TextAttributor()
        result = attributor.attribute(
            query_id="q_serial",
            dataset_id="test",
            pipeline_id="text-answer",
            pipeline_answer=None,
            pipeline_support_status="unsupported",
            gold_support_status="supported",
            oracle_results=dict.fromkeys(
                ["none", "R", "E", "A", "RE", "RA", "EA", "REA"], "supported"
            ),
        )
        d = result.to_dict()
        required_fields = [
            "query_id",
            "dataset_id",
            "pipeline_id",
            "pipeline_support_status",
            "gold_support_status",
            "total_error",
            "total_recoverable_error",
            "interaction_term",
            "phi_retrieval",
            "phi_extraction",
            "phi_aggregation",
            "positive_contributions",
            "negative_contributions",
            "dominant_fault",
            "dominant_fault_confidence",
            "failure_category",
            "value_function_note",
        ]
        for field in required_fields:
            assert field in d, f"Missing field in to_dict(): {field}"


# ---------------------------------------------------------------------------
# Backward compatibility: AttributionResult still has required fields
# ---------------------------------------------------------------------------


class TestAttributionResultBackwardCompat:
    def test_attribution_result_has_new_fields(self):
        """Check that the existing AttributionResult was updated with new fields."""
        from dataclasses import fields

        from faulttrace_pipelines.attribution import AttributionResult

        field_names = {f.name for f in fields(AttributionResult)}
        assert "total_recoverable_error" in field_names
        assert "positive_contributions" in field_names
        assert "negative_contributions" in field_names
        assert "value_function_note" in field_names

    def test_component_attribution_has_is_negative_field(self):
        from faulttrace_pipelines.attribution import ComponentAttribution

        ca = ComponentAttribution(component="scope", shapley_value=-0.2, is_negative=True)
        assert ca.is_negative is True
        d = ca.to_dict()
        assert "is_negative" in d
        assert d["is_negative"] is True
