"""
Intervention Execution Engine.

Builds and executes an OracleLatticeRunner that takes an existing completed
pipeline run and executes all eight intervention subsets of {R, E, A} using Oracles.
"""

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from faulttrace_core.models import GoldAnswer, PipelineRun, QuerySpec
from faulttrace_gold.oracles import AggregationOracle, ExtractionOracle, ScopeOracle
from pydantic import BaseModel

from faulttrace_pipelines.loss import LossDiagnostic, compute_loss


class LatticeRun(BaseModel):
    intervention_id: str
    parent_run_id: str
    subset: str  # "none", "R", "E", "A", "RE", "RA", "EA", "REA"
    answer_value: Any
    loss_diagnostic: LossDiagnostic
    status: str
    changed_records_count: int | None = None
    natural_language_summary: str | None = None


class LatticeDiagnosticSummary(BaseModel):
    parent_run_id: str
    baseline_loss: float
    subset_runs: dict[str, LatticeRun]
    phi_R: float
    phi_E: float
    phi_A: float
    interaction: float


class OracleLatticeRunner:
    """Executes all 8 counterfactual subsets for a given pipeline run."""

    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        self.artifacts_dir = artifacts_dir
        self.scope_oracle = ScopeOracle()
        self.extraction_oracle = ExtractionOracle()
        self.aggregation_oracle = AggregationOracle()

    def execute_lattice(
        self,
        parent_run: PipelineRun,
        query: QuerySpec,
        gold_answer: GoldAnswer,
        corpus_df: pd.DataFrame,
    ) -> LatticeDiagnosticSummary:
        """Execute the 8 subsets and return the full Shapley diagnostic."""
        subsets = ["none", "R", "E", "A", "RE", "RA", "EA", "REA"]
        subset_runs = {}

        for subset in subsets:
            replace_R = "R" in subset
            replace_E = "E" in subset
            replace_A = "A" in subset

            # Execute intervention
            lattice_run = self._execute_intervention(
                parent_run=parent_run,
                query=query,
                gold_answer=gold_answer,
                corpus_df=corpus_df,
                replace_R=replace_R,
                replace_E=replace_E,
                replace_A=replace_A,
                subset_name=subset,
            )
            subset_runs[subset] = lattice_run

        invalid = [name for name, run in subset_runs.items() if run.status != "valid"]
        if invalid:
            reasons = {
                name: subset_runs[name].natural_language_summary for name in invalid
            }
            raise ValueError(
                "Counterfactual attribution requires a complete valid lattice; "
                f"invalid subsets={reasons}"
            )

        # Compute exact Shapley
        baseline_loss = subset_runs["none"].loss_diagnostic.normalized_loss

        def v(subset_name: str) -> float:
            """
            Value function: reduction in loss when replacing subset 'subset_name' with oracle.

            Formula: v(S) = baseline_loss - loss(S)

            IMPORTANT: v(S) CAN be negative. This happens when an oracle replacement
            makes the answer WORSE (e.g. replacing a lucky-correct component with oracle
            reveals that oracle scope changes the aggregation domain in a harmful way).
            Negative values represent harmful oracle components and are preserved — not clamped.
            """
            loss = subset_runs[subset_name].loss_diagnostic.normalized_loss
            # Not clamped: can be negative if oracle replacement makes things worse
            return baseline_loss - loss

        # phi_i = sum_{S not containing i} weight(|S|) * (v(S ∪ {i}) - v(S))
        # For 3 components R, E, A:
        # weight(|S|=0) = 1/3, weight(|S|=1) = 1/6 per subset, weight(|S|=2) = 1/3
        def shapley(i: str, with_i: list[str], without_i: list[str]) -> float:
            total = 0.0

            # |S|=0: weight = 1/3
            total += (1 / 3) * (v(with_i[0]) - v("none"))

            # |S|=1: weight = 1/6 each (2 subsets of size 1 not containing i)
            total += (1 / 6) * (v(with_i[1]) - v(without_i[1]))
            total += (1 / 6) * (v(with_i[2]) - v(without_i[2]))

            # |S|=2: weight = 1/3
            total += (1 / 3) * (v("REA") - v(without_i[3]))

            # NOT clamped to 0: negative Shapley values are valid and informative
            return total

        phi_R = shapley("R", ["R", "RE", "RA", "REA"], ["none", "E", "A", "EA"])
        phi_E = shapley("E", ["E", "RE", "EA", "REA"], ["none", "R", "A", "RA"])
        phi_A = shapley("A", ["A", "RA", "EA", "REA"], ["none", "R", "E", "RE"])

        # Exact Shapley efficiency leaves no separate residual interaction term.
        # Keep the field for schema compatibility and expose only numerical drift.
        recoverable = v("REA")
        attributed = phi_R + phi_E + phi_A
        interaction = recoverable - attributed
        if abs(interaction) < 1e-12:
            interaction = 0.0

        return LatticeDiagnosticSummary(
            parent_run_id=parent_run.run_id,
            baseline_loss=baseline_loss,
            subset_runs=subset_runs,
            phi_R=phi_R,
            phi_E=phi_E,
            phi_A=phi_A,
            interaction=interaction,
        )

    def _execute_intervention(
        self,
        parent_run: PipelineRun,
        query: QuerySpec,
        gold_answer: GoldAnswer,
        corpus_df: pd.DataFrame,
        replace_R: bool,
        replace_E: bool,
        replace_A: bool,
        subset_name: str,
    ) -> LatticeRun:
        """Executes a single subset replacement."""
        status = "valid"
        answer_value = None

        def artifact_path(*keys: str) -> Path | None:
            for key in keys:
                value = parent_run.artifact_references.get(key)
                if value and Path(value).exists():
                    return Path(value)
            return None

        # Resolve the concrete pipeline once so non-oracle components can be
        # replayed on upstream counterfactual outputs.
        from faulttrace_pipelines import get_pipeline

        pipeline = get_pipeline(
            parent_run.pipeline_id,
            self.artifacts_dir / "component_replay",
            provider_id=parent_run.provider_id,
        )
        pipeline.execution_seed = parent_run.execution_seed

        # Stage 1: Retrieval/Scope
        if replace_R:
            scope_res = self.scope_oracle.evaluate(query, corpus_df)
            current_record_ids = scope_res.record_ids
        else:
            # Load from parent run artifact
            try:
                # We need to simulate the parent pipeline's scope or load it.
                # If pipeline was P4/P5, it stored extraction.parquet which implies the scope
                # Or we can simply re-execute the pipeline's scope logic if we don't have scope artifact
                # For baseline, we just use the original answer if none are replaced.
                if not replace_E and not replace_A:
                    answer_value = parent_run.answer
                    return self._build_lattice_run(
                        parent_run, subset_name, answer_value, gold_answer, query, status
                    )

                # We need the parent's actual scope if we replace E or A but NOT R.
                # If extraction.parquet exists, those are the records extracted.
                # The prompt states: "Reuse compatible cached non-oracle components where valid"
                # If parent run didn't save extraction, we fail diagnostic.
                scope_path = artifact_path("scope_enumerate", "scope_output")
                extract_path = artifact_path("fact_extract", "extraction")
                source_path = scope_path or extract_path
                if source_path:
                    parent_df = pd.read_parquet(source_path)
                    current_record_ids = (
                        parent_df["record_id"].tolist() if "record_id" in parent_df.columns else []
                    )
                    if "record_id" not in parent_df.columns:
                        raise ValueError("Parent scope artifact does not contain record_id")
                else:
                    status = "invalid"
                    return self._build_lattice_run(
                        parent_run,
                        subset_name,
                        None,
                        gold_answer,
                        query,
                        status,
                        "parent scope artifact is unavailable",
                    )
            except Exception as exc:
                status = "invalid"
                return self._build_lattice_run(
                    parent_run,
                    subset_name,
                    None,
                    gold_answer,
                    query,
                    status,
                    f"could not load parent scope: {exc}",
                )

        # Stage 2: Extraction
        if replace_E:
            # We supply current_record_ids to the Extraction Oracle
            # Filter corpus to just these IDs
            supplied_df = corpus_df[corpus_df["record_id"].isin(current_record_ids)]
            ext_res = self.extraction_oracle.evaluate(query.fact_spec, supplied_df)
            extracted_rows = ext_res.fact_rows
        else:
            try:
                current_df = corpus_df[corpus_df["record_id"].isin(current_record_ids)].copy()
                if replace_R:
                    # Critical compositional behavior: R* feeds the original
                    # pipeline extractor E-hat, rather than filtering cached E.
                    extracted_rows = pipeline.replay_extraction(query, current_df)
                else:
                    extract_path = artifact_path("fact_extract", "extraction")
                    if not extract_path:
                        raise ValueError("parent extraction artifact is unavailable")
                    parent_df = pd.read_parquet(extract_path)
                    if "record_id" in parent_df.columns:
                        parent_df = parent_df[parent_df["record_id"].isin(current_record_ids)]
                    extracted_rows = parent_df.to_dict(orient="records")
            except Exception as exc:
                status = "invalid"
                return self._build_lattice_run(
                    parent_run,
                    subset_name,
                    None,
                    gold_answer,
                    query,
                    status,
                    f"pipeline extraction replay failed: {exc}",
                )

        # Stage 3: Aggregation
        if replace_A:
            agg_res = self.aggregation_oracle.evaluate(
                query.aggregation_spec, extracted_rows, query
            )
            answer_value = agg_res.answer_value
        else:
            answer_value = pipeline.replay_aggregation(query, extracted_rows)

        return self._build_lattice_run(
            parent_run, subset_name, answer_value, gold_answer, query, status
        )

    def _build_lattice_run(
        self,
        parent_run: PipelineRun,
        subset_name: str,
        answer_value: Any,
        gold_answer: GoldAnswer,
        query: QuerySpec,
        status: str,
        reason: str | None = None,
    ) -> LatticeRun:
        if status == "valid":
            loss_diag = compute_loss(
                answer_value,
                gold_answer.answer_value,
                query.aggregation_spec,
                gold_answer.tolerance,
            )
        else:
            loss_diag = LossDiagnostic(normalized_loss=1.0, status="invalid")

        summary = (
            f"Subset {subset_name} yielded normalized loss {loss_diag.normalized_loss:.4f}."
            if status == "valid"
            else f"Subset {subset_name} is invalid: {reason or 'component replay unavailable'}."
        )

        run_obj = LatticeRun(
            intervention_id=str(uuid4()),
            parent_run_id=parent_run.run_id,
            subset=subset_name,
            answer_value=answer_value,
            loss_diagnostic=loss_diag,
            status=status,
            natural_language_summary=summary,
        )

        # Save to disk
        int_dir = self.artifacts_dir / "interventions" / run_obj.intervention_id
        int_dir.mkdir(parents=True, exist_ok=True)

        run_file = int_dir / "intervention.json"
        run_file.write_text(run_obj.model_dump_json(indent=2), encoding="utf-8")

        return run_obj
