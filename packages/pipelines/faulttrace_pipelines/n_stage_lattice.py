"""
Generalized N-Stage Counterfactual Execution Engine.

Builds and executes a GeneralLatticeRunner that takes a completed
pipeline run, a defined DAG PipelineGraph, and computes exact Shapley
recoverable-error contributions for an arbitrary number of stages.
"""

import math
from itertools import chain, combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

from faulttrace_core.evaluation import StageOracle, StageReplayBoundary
from faulttrace_core.models import (
    CounterfactualWorld,
    InterventionSet,
    PipelineGraph,
    PipelineRun,
    RunStatus,
)
from pydantic import BaseModel


class StageExecutionDefinition(BaseModel):
    """Configuration for how a stage executes during a counterfactual run."""
    stage_id: str
    oracle: Any  # Should be StageOracle, typed Any for pydantic
    boundary: Any  # Should be StageReplayBoundary, typed Any for pydantic


class GeneralLatticeDiagnosticSummary(BaseModel):
    parent_run_id: str
    baseline_loss: float
    worlds: dict[frozenset[str], CounterfactualWorld]
    shapley_values: dict[str, float]
    interaction: float


class GeneralLatticeRunner:
    """Executes all 2^N counterfactual subsets for a DAG pipeline."""

    def __init__(self, artifacts_dir: Path = Path("artifacts/runs")):
        self.artifacts_dir = artifacts_dir

    def execute_lattice(
        self,
        parent_run: PipelineRun,
        graph: PipelineGraph,
        stage_definitions: dict[str, StageExecutionDefinition],
        baseline_loss: float,
        evaluate_world_fn: Any,  # Callable[[set[str]], CounterfactualWorld]
    ) -> GeneralLatticeDiagnosticSummary:
        """Execute the 2^N subsets and return the full Shapley diagnostic."""
        
        stages = list(graph.stages.keys())
        n = len(stages)
        if n > 12:
            raise ValueError(f"Exact Shapley computation is O(2^N). N={n} is too large.")

        # 1. Enumerate all subsets S
        all_subsets = list(self._powerset(stages))
        
        worlds: dict[frozenset[str], CounterfactualWorld] = {}
        
        # 2. Evaluate each world
        for subset in all_subsets:
            subset_key = frozenset(subset)
            world = evaluate_world_fn(set(subset))
            if world.status != RunStatus.COMPLETED:
                raise ValueError(
                    f"Counterfactual attribution requires a complete valid lattice; "
                    f"subset {set(subset)} failed: {world.natural_language_summary}"
                )
            worlds[subset_key] = world

        # 3. Compute Value function v(S)
        def v(s: frozenset[str]) -> float:
            loss = worlds[s].loss_diagnostic.normalized_loss
            return baseline_loss - loss

        # 4. Compute Shapley values
        shapley_values = {stage: 0.0 for stage in stages}
        
        for i in stages:
            # subsets not containing i
            others = [x for x in stages if x != i]
            for subset in self._powerset(others):
                s = frozenset(subset)
                s_union_i = s | frozenset([i])
                
                size_s = len(s)
                # Weight: |S|! (N - |S| - 1)! / N!
                weight = (math.factorial(size_s) * math.factorial(n - size_s - 1)) / math.factorial(n)
                
                marginal = v(s_union_i) - v(s)
                shapley_values[i] += weight * marginal

        # 5. Compute interaction (residual)
        # Exact Shapley efficiency should mean interaction is near 0.
        full_set = frozenset(stages)
        recoverable = v(full_set)
        attributed = sum(shapley_values.values())
        interaction = recoverable - attributed
        
        if abs(interaction) < 1e-12:
            interaction = 0.0

        return GeneralLatticeDiagnosticSummary(
            parent_run_id=parent_run.run_id,
            baseline_loss=baseline_loss,
            worlds=worlds,
            shapley_values=shapley_values,
            interaction=interaction,
        )

    def _powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
