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

This module delegates execution to the OracleLatticeRunner and formats the exact 
Shapley values into AttributionResult.
"""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd
from dataclasses import dataclass, field

from faulttrace_core.models import GoldAnswer, QuerySpec, PipelineRun
from faulttrace_pipelines.lattice import OracleLatticeRunner


@dataclass
class ComponentAttribution:
    """Attribution result for a single pipeline component."""
    component: str          # "scope" | "facts" | "aggregation"
    shapley_value: float    # Shapley attribution ∈ [0, 1]

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "shapley_value": round(self.shapley_value, 6)
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

    total_error: float      # Normalized absolute error ∈ [0, +∞]
    interaction_term: float # Residual not explained by any single component

    components: list[ComponentAttribution] = field(default_factory=list)

    # Dominant fault component (highest shapley value)
    dominant_fault: Optional[str] = None
    dominant_fault_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_answer": self.pipeline_answer,
            "gold_answer": self.gold_answer,
            "is_correct": self.is_correct,
            "total_error": round(self.total_error, 6),
            "interaction_term": round(self.interaction_term, 6),
            "dominant_fault": self.dominant_fault,
            "dominant_fault_confidence": round(self.dominant_fault_confidence, 4),
            "components": [c.to_dict() for c in self.components],
        }


class CounterfactualAttributor:
    """
    Computes exact Shapley-inspired attribution for pipeline fault localization
    using the OracleLatticeRunner.
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
        Run exact lattice evaluation and compute attribution.
        """
        # 1. Execute lattice
        lattice_summary = self.lattice_runner.execute_lattice(
            parent_run=parent_run,
            query=query,
            gold_answer=gold_answer_obj,
            corpus_df=oracle_df
        )

        # 2. Extract values
        cf_none = lattice_summary.subset_runs["none"]
        baseline_loss = lattice_summary.baseline_loss

        # If it was fully correct to begin with, zero out attribution
        if baseline_loss == 0.0:
            phi_R = phi_E = phi_A = interaction = 0.0
            dominant = "none"
            dominant_conf = 0.0
        else:
            phi_R = lattice_summary.phi_R
            phi_E = lattice_summary.phi_E
            phi_A = lattice_summary.phi_A
            interaction = lattice_summary.interaction

            comps = {"scope": phi_R, "facts": phi_E, "aggregation": phi_A}
            dominant = max(comps, key=comps.get)
            dominant_conf = comps[dominant]

        components = [
            ComponentAttribution(component="scope", shapley_value=phi_R),
            ComponentAttribution(component="facts", shapley_value=phi_E),
            ComponentAttribution(component="aggregation", shapley_value=phi_A),
        ]

        # Check if pipeline agreed completely with baseline subset execution
        # (This is true if it was evaluated as valid)
        is_correct = parent_run.is_correct if parent_run.is_correct is not None else (baseline_loss == 0.0)

        return AttributionResult(
            run_id=parent_run.run_id,
            query_id=str(query.query_id),
            pipeline_id=parent_run.pipeline_id,
            pipeline_answer=parent_run.answer,
            gold_answer=gold_answer_obj.answer_value,
            is_correct=is_correct,
            total_error=baseline_loss,
            interaction_term=interaction,
            components=components,
            dominant_fault=dominant,
            dominant_fault_confidence=dominant_conf,
        )
