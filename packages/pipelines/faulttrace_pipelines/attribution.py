"""
Counterfactual Fault Attribution Engine.

Implements oracle-replacement attribution for the three error sources:
  R — Evidence Scope  (which records are retrieved)
  E — Fact Extraction (which field values are extracted)
  A — Aggregation     (how the facts are reduced to an answer)

Attribution Algorithm (Shapley-inspired oracle replacement):
─────────────────────────────────────────────────────────────
Given:
  pipeline_answer  = pipeline(R_hat, E_hat, A_hat)
  gold_answer      = oracle(R*, E*, A*)

We define 8 counterfactual runs replacing subsets of {R, E, A} with oracle:
  cf(none) = pipeline(R_hat, E_hat, A_hat)
  cf(R)    = pipeline(R*,   E_hat, A_hat)
  cf(E)    = pipeline(R_hat, E*,  A_hat)
  cf(A)    = pipeline(R_hat, E_hat, A*)
  cf(RE)   = pipeline(R*,   E*,   A_hat)
  cf(RA)   = pipeline(R*,   E_hat, A*)
  cf(EA)   = pipeline(R_hat, E*,   A*)
  cf(REA)  = pipeline(R*,   E*,   A*)  ≡ gold

This module delegates component-faithful replay to OracleLatticeRunner and
formats Shapley values only when all eight interventions are valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from faulttrace_core.models import GoldAnswer, PipelineRun, QuerySpec
from faulttrace_gold.oracles import AggregationOracle, ExtractionOracle, ScopeOracle
from faulttrace_gold.validator import results_agree

from faulttrace_pipelines.lattice import OracleLatticeRunner


@dataclass
class ComponentAttribution:
    """Attribution result for a single pipeline component."""

    component: str  # "scope" | "facts" | "aggregation"
    shapley_value: float  # Shapley attribution. CAN be negative (harmful oracle component).
    is_negative: bool = False  # True when oracle replacement made the answer WORSE

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "shapley_value": round(self.shapley_value, 6),
            "is_negative": self.is_negative,
        }


@dataclass
class AttributionResult:
    """Full attribution result for a pipeline run."""

    run_id: str
    query_id: str
    pipeline_id: str

    pipeline_answer: Any
    gold_answer: Any
    is_correct: bool

    total_error: float  # Normalized loss at baseline ∈ [0, +∞]
    total_recoverable_error: float  # v(REA): max error recoverable by oracle replacement
    interaction_term: float  # Deprecated numerical residual; zero for an exact complete lattice.

    # Reported separately per the efficiency axiom:
    # phi_R + phi_E + phi_A + interaction_term = total_recoverable_error
    positive_contributions: list[str] = field(default_factory=list)  # components with phi > 0
    negative_contributions: list[str] = field(default_factory=list)  # components with phi < 0

    components: list[ComponentAttribution] = field(default_factory=list)
    interventions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Dominant fault component (highest |shapley value|)
    dominant_fault: str | None = None
    dominant_fault_confidence: float = 0.0

    # These sets answer different questions. Outcome-active components have a
    # non-negligible Shapley effect on answer loss. Artifact discrepancies are
    # direct offline comparisons with the corresponding oracle component and
    # can expose an injected fault that is masked downstream.
    outcome_active_faults: list[str] = field(default_factory=list)
    artifact_discrepancy_faults: list[str] = field(default_factory=list)
    component_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)

    value_function_note: str = (
        "v(S) = baseline_loss - loss(S). "
        "Shapley weights: |S|=0: 1/3, |S|=1: 1/6, |S|=2: 1/3. "
        "Loss is normalized absolute error for scalars, Jaccard distance for TopK. "
        "Negative phi indicates oracle replacement worsens the answer. "
        "NOT clamped to [0,1]. A complete exact Shapley lattice has zero residual."
    )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_answer": self.pipeline_answer,
            "gold_answer": self.gold_answer,
            "is_correct": self.is_correct,
            "total_error": round(self.total_error, 6),
            "total_recoverable_error": round(self.total_recoverable_error, 6),
            "interaction_term": round(self.interaction_term, 6),
            "positive_contributions": self.positive_contributions,
            "negative_contributions": self.negative_contributions,
            "dominant_fault": self.dominant_fault,
            "dominant_fault_confidence": round(self.dominant_fault_confidence, 4),
            "outcome_active_faults": self.outcome_active_faults,
            "artifact_discrepancy_faults": self.artifact_discrepancy_faults,
            "predicted_faults": self.artifact_discrepancy_faults,
            "component_diagnostics": self.component_diagnostics,
            "components": [c.to_dict() for c in self.components],
            "interventions": self.interventions,
            "value_function_note": self.value_function_note,
            "multilabel_diagnostic_note": (
                "outcome_active_faults uses |Shapley value| > 1e-9; "
                "artifact_discrepancy_faults compares persisted component artifacts "
                "with R*/E*/A* offline. Artifact diagnosis requires oracle access and "
                "is not an online certificate or a claim of unique real-world causality."
            ),
        }


def _artifact_path(parent_run: PipelineRun, *keys: str) -> Path | None:
    for key in keys:
        value = parent_run.artifact_references.get(key)
        if value and Path(value).exists():
            return Path(value)
    return None


def _canonical_fact_rows(
    frame: pd.DataFrame, query: QuerySpec
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return order-invariant fact rows restricted to oracle-comparable fields."""
    comparable = ["record_id", *query.fact_spec.fields]
    comparable = list(dict.fromkeys(field for field in comparable if field in frame.columns))
    if "record_id" not in comparable:
        return [], comparable
    selected = frame[comparable].copy()
    selected["record_id"] = selected["record_id"].astype(str)
    selected = selected.sort_values(comparable, key=lambda col: col.astype(str), kind="stable")
    return selected.to_dict(orient="records"), comparable


def _fact_rows_agree(
    observed: pd.DataFrame, expected: pd.DataFrame, query: QuerySpec
) -> tuple[bool | None, dict[str, Any]]:
    required = list(dict.fromkeys(["record_id", *query.fact_spec.fields]))
    if observed.empty and expected.empty:
        missing = sorted(set(required) - set(observed.columns))
        details: dict[str, Any] = {
            "observed_row_count": 0,
            "oracle_row_count": 0,
            "compared_fields": [field for field in required if field in observed.columns],
        }
        if missing:
            details["missing_comparable_fields"] = missing
        return not missing, details
    observed_rows, observed_fields = _canonical_fact_rows(observed, query)
    expected_rows, expected_fields = _canonical_fact_rows(expected, query)
    fields = [field for field in expected_fields if field in observed_fields]
    details: dict[str, Any] = {
        "observed_row_count": len(observed_rows),
        "oracle_row_count": len(expected_rows),
        "compared_fields": fields,
    }
    if set(required) - set(fields):
        details["missing_comparable_fields"] = sorted(set(required) - set(fields))
        return False, details
    if len(observed_rows) != len(expected_rows):
        return False, details
    def values_agree(observed_value: Any, expected_value: Any) -> bool:
        both_missing = (
            pd.api.types.is_scalar(observed_value)
            and pd.api.types.is_scalar(expected_value)
            and bool(pd.isna(observed_value))
            and bool(pd.isna(expected_value))
        )
        return both_missing or results_agree(observed_value, expected_value, query.tolerance)

    agrees = all(
        values_agree(observed_row[field], expected_row[field])
        for observed_row, expected_row in zip(observed_rows, expected_rows, strict=True)
        for field in fields
    )
    return agrees, details


def diagnose_component_artifacts(
    parent_run: PipelineRun, query: QuerySpec, corpus_df: pd.DataFrame
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Compare R/E/A artifacts with their stage-matched oracle outputs.

    This is an offline diagnostic. Each comparison holds the component input
    fixed: E* receives the observed scope and A* receives the observed facts.
    Consequently, upstream faults are not double-counted as downstream faults.
    """
    diagnostics: dict[str, dict[str, Any]] = {}
    discrepancies: list[str] = []
    scope_path = _artifact_path(parent_run, "scope_enumerate", "scope_output")
    extraction_path = _artifact_path(parent_run, "fact_extract", "extraction")

    observed_scope_ids: list[str] | None = None
    if scope_path is None:
        diagnostics["scope"] = {"status": "unknown", "reason": "scope artifact unavailable"}
    else:
        try:
            scope_frame = pd.read_parquet(scope_path, columns=["record_id"])
            observed_scope_ids = scope_frame["record_id"].astype(str).tolist()
            oracle_scope_ids = [
                str(value) for value in ScopeOracle().evaluate(query, corpus_df).record_ids
            ]
            agrees = sorted(observed_scope_ids) == sorted(oracle_scope_ids)
            diagnostics["scope"] = {
                "status": "match" if agrees else "mismatch",
                "observed_count": len(observed_scope_ids),
                "oracle_count": len(oracle_scope_ids),
            }
            if not agrees:
                discrepancies.append("scope")
        except Exception as exc:
            diagnostics["scope"] = {"status": "unknown", "reason": str(exc)}

    observed_facts: pd.DataFrame | None = None
    if extraction_path is None or observed_scope_ids is None:
        diagnostics["facts"] = {
            "status": "unknown",
            "reason": "extraction artifact or observed scope unavailable",
        }
    else:
        try:
            observed_facts = pd.read_parquet(extraction_path)
            supplied = corpus_df[
                corpus_df["record_id"].astype(str).isin(set(observed_scope_ids))
            ]
            oracle_rows = ExtractionOracle().evaluate(query.fact_spec, supplied).fact_rows
            agrees, details = _fact_rows_agree(observed_facts, pd.DataFrame(oracle_rows), query)
            diagnostics["facts"] = {
                "status": "match" if agrees else "mismatch" if agrees is False else "unknown",
                **details,
            }
            if agrees is False:
                discrepancies.append("facts")
        except Exception as exc:
            diagnostics["facts"] = {"status": "unknown", "reason": str(exc)}

    if observed_facts is None:
        diagnostics["aggregation"] = {
            "status": "unknown",
            "reason": "extraction artifact unavailable",
        }
    else:
        try:
            aggregation_input = observed_facts
            if "scope_decision" in aggregation_input.columns:
                aggregation_input = aggregation_input[
                    aggregation_input["scope_decision"] == "in_scope"
                ]
            oracle_answer = AggregationOracle().evaluate(
                query.aggregation_spec,
                aggregation_input.to_dict(orient="records"),
                query,
            ).answer_value
            agrees = results_agree(parent_run.answer, oracle_answer, query.tolerance)
            diagnostics["aggregation"] = {
                "status": "match" if agrees else "mismatch",
                "oracle_replay_answer": oracle_answer,
            }
            if not agrees:
                discrepancies.append("aggregation")
        except Exception as exc:
            diagnostics["aggregation"] = {"status": "unknown", "reason": str(exc)}

    return discrepancies, diagnostics


class CounterfactualAttributor:
    """
    Computes Shapley attribution over a complete component-replay lattice.
    """

    def __init__(self):
        self.lattice_runner = OracleLatticeRunner()

    def attribute(
        self,
        parent_run: PipelineRun,
        query: QuerySpec,
        gold_answer_obj: GoldAnswer,
        oracle_df: pd.DataFrame,
    ) -> AttributionResult:
        """
        Run the complete lattice and compute attribution.
        """
        # 1. Execute lattice
        lattice_summary = self.lattice_runner.execute_lattice(
            parent_run=parent_run, query=query, gold_answer=gold_answer_obj, corpus_df=oracle_df
        )

        # 2. Extract values
        lattice_summary.subset_runs["none"]
        baseline_loss = lattice_summary.baseline_loss

        # If it was fully correct to begin with, zero out attribution
        if baseline_loss == 0.0:
            phi_R = phi_E = phi_A = interaction = 0.0
            recoverable = 0.0
            dominant = "none"
            dominant_conf = 0.0
        else:
            phi_R = lattice_summary.phi_R
            phi_E = lattice_summary.phi_E
            phi_A = lattice_summary.phi_A
            interaction = lattice_summary.interaction
            recoverable = phi_R + phi_E + phi_A + interaction

            # Dominant is by ABSOLUTE value (negative phi is still informative)
            comps = {"scope": phi_R, "facts": phi_E, "aggregation": phi_A}
            dominant = max(comps, key=lambda k: abs(comps[k]))
            dominant_conf = abs(comps[dominant])

        positive_contribs = [
            name
            for name, phi in [("scope", phi_R), ("facts", phi_E), ("aggregation", phi_A)]
            if phi > 0
        ]
        negative_contribs = [
            name
            for name, phi in [("scope", phi_R), ("facts", phi_E), ("aggregation", phi_A)]
            if phi < 0
        ]

        components = [
            ComponentAttribution(component="scope", shapley_value=phi_R, is_negative=phi_R < 0),
            ComponentAttribution(component="facts", shapley_value=phi_E, is_negative=phi_E < 0),
            ComponentAttribution(
                component="aggregation", shapley_value=phi_A, is_negative=phi_A < 0
            ),
        ]

        is_correct = (
            parent_run.is_correct if parent_run.is_correct is not None else (baseline_loss == 0.0)
        )
        outcome_active_faults = [
            name
            for name, phi in [("scope", phi_R), ("facts", phi_E), ("aggregation", phi_A)]
            if abs(phi) > 1e-9
        ]
        artifact_discrepancies, component_diagnostics = diagnose_component_artifacts(
            parent_run, query, oracle_df
        )

        return AttributionResult(
            run_id=parent_run.run_id,
            query_id=str(query.query_id),
            pipeline_id=parent_run.pipeline_id,
            pipeline_answer=parent_run.answer,
            gold_answer=gold_answer_obj.answer_value,
            is_correct=is_correct,
            total_error=baseline_loss,
            total_recoverable_error=recoverable,
            interaction_term=interaction,
            positive_contributions=positive_contribs,
            negative_contributions=negative_contribs,
            components=components,
            interventions={
                name.lower(): {
                    "answer_value": run.answer_value,
                    "normalized_loss": run.loss_diagnostic.normalized_loss,
                    "status": run.status,
                }
                for name, run in lattice_summary.subset_runs.items()
            },
            dominant_fault=dominant,
            dominant_fault_confidence=dominant_conf,
            outcome_active_faults=outcome_active_faults,
            artifact_discrepancy_faults=artifact_discrepancies,
            component_diagnostics=component_diagnostics,
        )
