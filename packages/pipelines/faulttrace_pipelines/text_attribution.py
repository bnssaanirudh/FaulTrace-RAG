"""
Text-Benchmark Attribution Engine for FaultTrace-RAG.

Extends the oracle-lattice attribution framework to text-benchmark datasets
(SciFact, RAGBench, HotpotQA) where the "answer" is a CitedAnswer with a
SupportStatus rather than a numeric value.

Value function for text attribution:
  v(S) = baseline_loss - loss(S)
  loss = f(support_status):
    SUPPORTED = 0.0
    PARTIALLY_SUPPORTED = 0.5
    UNSUPPORTED = 1.0
    CONFLICTING = 0.8
    INSUFFICIENT_EVIDENCE = 0.5 (neither confirmed nor refuted)

R/E/A components in text benchmarks:
  R (Retrieval): which documents are retrieved and ranked
  E (Extraction): which facts, spans, and citations are produced from those docs
  A (Aggregation): how facts are combined into the final support_status assessment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Benchmark Error Taxonomy
# ---------------------------------------------------------------------------


class RetrievalError(StrEnum):
    """Failure modes in the Retrieval (R) component."""

    MISSED_RELEVANT_EVIDENCE = "missed_relevant_evidence"
    LOW_RANK = "low_rank"
    IRRELEVANT_CONTEXT = "irrelevant_context"
    DUPLICATE_CONTEXT = "duplicate_context"
    TRUNCATION = "truncation"


class ExtractionError(StrEnum):
    """Failure modes in the Extraction (E) component."""

    MISSED_FACT = "missed_fact"
    WRONG_VALUE = "wrong_value"
    WRONG_ENTITY = "wrong_entity"
    WRONG_RELATION = "wrong_relation"
    INVALID_SPAN = "invalid_span"
    HALLUCINATED_CITATION = "hallucinated_citation"


class AggregationError(StrEnum):
    """Failure modes in the Aggregation/Reasoning (A) component."""

    WRONG_COUNT = "wrong_count"
    WRONG_COMPARISON = "wrong_comparison"
    DENOMINATOR_ERROR = "denominator_error"
    DUPLICATE_AGGREGATION = "duplicate_aggregation"
    UNIT_DATE_ERROR = "unit_date_error"
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"


class MixedError(StrEnum):
    """Compound error modes spanning multiple components."""

    R_PLUS_E = "R+E"
    R_PLUS_A = "R+A"
    E_PLUS_A = "E+A"
    R_PLUS_E_PLUS_A = "R+E+A"


class BenchmarkErrorTaxonomy:
    """
    Container for all benchmark error taxonomy enums.

    Provides a single import point for the full failure taxonomy:
      - BenchmarkErrorTaxonomy.Retrieval: retrieval-stage failure modes
      - BenchmarkErrorTaxonomy.Extraction: extraction-stage failure modes
      - BenchmarkErrorTaxonomy.Aggregation: aggregation/reasoning failure modes
      - BenchmarkErrorTaxonomy.Mixed: compound multi-component failure modes
    """

    Retrieval = RetrievalError
    Extraction = ExtractionError
    Aggregation = AggregationError
    Mixed = MixedError


@dataclass
class BenchmarkFailureCase:
    """
    A labeled failure case for benchmark attribution evaluation.

    These are used as ground-truth labels to validate that the attribution
    engine correctly identifies which component caused the failure.
    """

    case_id: str
    dataset_id: str
    query_id: str
    expected_dominant_fault: str  # "retrieval" | "extraction" | "aggregation" | "none"
    expected_error_type: str | None = None  # from taxonomy enums above
    gold_support_status: str = "supported"  # from SupportStatus
    description: str = ""


@dataclass
class TextAttributionResult:
    """
    Attribution result for a single text-benchmark query.

    Unlike numeric attribution (CounterfactualAttributor), this uses
    SupportStatus-based loss values. Shapley values are NOT clamped to [0,1].
    """

    query_id: str
    dataset_id: str
    pipeline_id: str

    pipeline_support_status: str
    gold_support_status: str
    pipeline_answer_text: str | None

    total_error: float  # loss at baseline
    total_recoverable_error: float  # v(REA)
    interaction_term: float  # v(REA) - sum(phi_i), can be negative

    phi_retrieval: float  # Shapley for R
    phi_extraction: float  # Shapley for E
    phi_aggregation: float  # Shapley for A

    positive_contributions: list[str] = field(default_factory=list)
    negative_contributions: list[str] = field(default_factory=list)

    dominant_fault: str | None = None
    dominant_fault_confidence: float = 0.0

    failure_category: str | None = None  # From BenchmarkErrorTaxonomy

    value_function_note: str = (
        "v(S) = baseline_loss - loss(S). "
        "loss(support_status): SUPPORTED=0.0, PARTIALLY=0.5, INSUFFICIENT=0.5, "
        "CONFLICTING=0.8, UNSUPPORTED=1.0. "
        "NOT clamped. interaction = v(REA) - sum(phi_i)."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "dataset_id": self.dataset_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_support_status": self.pipeline_support_status,
            "gold_support_status": self.gold_support_status,
            "total_error": round(self.total_error, 6),
            "total_recoverable_error": round(self.total_recoverable_error, 6),
            "interaction_term": round(self.interaction_term, 6),
            "phi_retrieval": round(self.phi_retrieval, 6),
            "phi_extraction": round(self.phi_extraction, 6),
            "phi_aggregation": round(self.phi_aggregation, 6),
            "positive_contributions": self.positive_contributions,
            "negative_contributions": self.negative_contributions,
            "dominant_fault": self.dominant_fault,
            "dominant_fault_confidence": round(self.dominant_fault_confidence, 4),
            "failure_category": self.failure_category,
            "value_function_note": self.value_function_note,
        }


# ---------------------------------------------------------------------------
# Support Status Loss
# ---------------------------------------------------------------------------


def support_status_loss(status: str) -> float:
    """
    Map a SupportStatus string to a loss value in [0.0, 1.0].

    This is the value function used by TextAttributor. The mapping is:
    - SUPPORTED             → 0.0 (perfect)
    - PARTIALLY_SUPPORTED   → 0.5 (half credit)
    - INSUFFICIENT_EVIDENCE → 0.5 (abstention, neutral)
    - CONFLICTING           → 0.8 (bad — active conflict found)
    - UNSUPPORTED           → 1.0 (worst — document refutes the claim)
    """
    mapping = {
        "supported": 0.0,
        "partially_supported": 0.5,
        "insufficient_evidence": 0.5,
        "conflicting": 0.8,
        "unsupported": 1.0,
    }
    return mapping.get(status.lower(), 0.5)


# ---------------------------------------------------------------------------
# TextAttributor
# ---------------------------------------------------------------------------


class TextAttributor:
    """
    Counterfactual attribution engine for text-benchmark pipelines.

    Uses the same 8-subset oracle-lattice as CounterfactualAttributor, but:
    - The value function is based on SupportStatus, not numeric error
    - Requires a callable for each oracle component (R*, E*, A*)
    - Negative Shapley values are preserved and reported
    """

    def attribute(
        self,
        query_id: str,
        dataset_id: str,
        pipeline_id: str,
        pipeline_answer: Any | None,
        pipeline_support_status: str,
        gold_support_status: str,
        oracle_results: dict[str, str],  # subset -> support_status from oracle
    ) -> TextAttributionResult:
        """
        Compute attribution given oracle subset results.

        oracle_results: dict mapping subset name ('none', 'R', 'E', 'A', 'RE', 'RA', 'EA', 'REA')
                        to the SupportStatus string that would have been produced.
        """

        def v(subset: str) -> float:
            baseline = support_status_loss(pipeline_support_status)
            if subset not in oracle_results:
                return 0.0
            subset_loss = support_status_loss(oracle_results[subset])
            return baseline - subset_loss  # Can be negative

        baseline_loss = support_status_loss(pipeline_support_status)

        if baseline_loss == 0.0:
            phi_r = phi_e = phi_a = interaction = recoverable = 0.0
            dominant = "none"
            dominant_conf = 0.0
        else:
            phi_r = self._shapley("R", v, ["R", "RE", "RA", "REA"], ["none", "E", "A", "EA"])
            phi_e = self._shapley("E", v, ["E", "RE", "EA", "REA"], ["none", "R", "A", "RA"])
            phi_a = self._shapley("A", v, ["A", "RA", "EA", "REA"], ["none", "R", "E", "RE"])

            recoverable = v("REA")
            interaction = recoverable - (phi_r + phi_e + phi_a)

            comps = {"retrieval": phi_r, "extraction": phi_e, "aggregation": phi_a}
            dominant = max(comps, key=lambda k: abs(comps[k]))
            dominant_conf = abs(comps[dominant])

        positive = [
            n
            for n, phi in [("retrieval", phi_r), ("extraction", phi_e), ("aggregation", phi_a)]
            if phi > 0
        ]
        negative = [
            n
            for n, phi in [("retrieval", phi_r), ("extraction", phi_e), ("aggregation", phi_a)]
            if phi < 0
        ]

        # Classify failure category
        failure_cat = self._classify_failure(phi_r, phi_e, phi_a)

        return TextAttributionResult(
            query_id=query_id,
            dataset_id=dataset_id,
            pipeline_id=pipeline_id,
            pipeline_support_status=pipeline_support_status,
            gold_support_status=gold_support_status,
            pipeline_answer_text=str(pipeline_answer) if pipeline_answer else None,
            total_error=baseline_loss,
            total_recoverable_error=recoverable,
            interaction_term=interaction,
            phi_retrieval=phi_r,
            phi_extraction=phi_e,
            phi_aggregation=phi_a,
            positive_contributions=positive,
            negative_contributions=negative,
            dominant_fault=dominant,
            dominant_fault_confidence=dominant_conf,
            failure_category=failure_cat,
        )

    @staticmethod
    def _shapley(
        component: str,
        v,
        with_i: list[str],
        without_i: list[str],
    ) -> float:
        """Exact Shapley value for 3 components. NOT clamped."""
        total = 0.0
        total += (1 / 3) * (v(with_i[0]) - v("none"))
        total += (1 / 6) * (v(with_i[1]) - v(without_i[1]))
        total += (1 / 6) * (v(with_i[2]) - v(without_i[2]))
        total += (1 / 3) * (v("REA") - v(without_i[3]))
        return total

    @staticmethod
    def _classify_failure(phi_r: float, phi_e: float, phi_a: float) -> str:
        """Classify the failure into the error taxonomy."""
        threshold = 0.05
        has_r = abs(phi_r) > threshold
        has_e = abs(phi_e) > threshold
        has_a = abs(phi_a) > threshold

        if has_r and has_e and has_a:
            return MixedError.R_PLUS_E_PLUS_A.value
        elif has_r and has_e:
            return MixedError.R_PLUS_E.value
        elif has_r and has_a:
            return MixedError.R_PLUS_A.value
        elif has_e and has_a:
            return MixedError.E_PLUS_A.value
        elif has_r:
            return "retrieval"
        elif has_e:
            return "extraction"
        elif has_a:
            return "aggregation"
        else:
            return "none"
