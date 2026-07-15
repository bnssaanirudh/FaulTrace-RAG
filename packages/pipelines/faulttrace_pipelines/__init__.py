"""
faulttrace_pipelines — pipeline implementations and attribution engine.

Exported pipelines:
  P0DeterministicBaseline  — correct oracle (baseline)
  P1WrongScope             — faulty scope, correct E and A
  P2WrongFacts             — correct scope, faulty E, correct A
  P3WrongAggregation       — correct scope and E, faulty A
  P4CompoundSF             — faulty scope and E, correct A
  P5FullCompound           — all three faulted

Attribution:
  CounterfactualAttributor — Shapley oracle-replacement engine
  AttributionResult        — full attribution result
  ComponentAttribution     — per-component attribution
"""

from faulttrace_pipelines.base import AbstractPipeline
from faulttrace_pipelines.p0_baseline import P0DeterministicBaseline
from faulttrace_pipelines.p1_wrong_scope import P1WrongScope
from faulttrace_pipelines.p2_wrong_facts import P2WrongFacts
from faulttrace_pipelines.p3_wrong_aggregation import P3WrongAggregation
from faulttrace_pipelines.p4_compound_sf import P4CompoundSF
from faulttrace_pipelines.p5_full_compound import P5FullCompound
from faulttrace_pipelines.p4_full_scope_mer import P4FullScopeMERPipeline
from faulttrace_pipelines.p5_certified_mer import P5CertifiedMERPipeline
from faulttrace_pipelines.p1_direct_bm25 import P1DirectBM25Pipeline
from faulttrace_pipelines.p2_direct_dense import P2DirectDensePipeline
from faulttrace_pipelines.p3_extract_aggregate import P3ExtractAggregatePipeline
from faulttrace_pipelines.attribution import (
    CounterfactualAttributor,
    AttributionResult,
    ComponentAttribution,
)
from faulttrace_pipelines.query_factory import QueryFactory

# Pipeline registry — maps pipeline_id → class
PIPELINE_REGISTRY: dict[str, type[AbstractPipeline]] = {
    "P0-deterministic-scope-baseline": P0DeterministicBaseline,
    "P1-wrong-scope": P1WrongScope,
    "P2-wrong-facts": P2WrongFacts,
    "P3-wrong-aggregation": P3WrongAggregation,
    "P4-compound-scope-facts": P4CompoundSF,
    "P5-full-compound": P5FullCompound,
    # Research pipelines added in Prompt 3
    "P1-direct-bm25": P1DirectBM25Pipeline,
    "P2-direct-dense": P2DirectDensePipeline,
    "P3-extract-aggregate": P3ExtractAggregatePipeline,
    "P4-full-scope-mer": P4FullScopeMERPipeline,
    "P5-certified-mer-repair": P5CertifiedMERPipeline,
}


def get_pipeline(pipeline_id: str, artifacts_dir=None) -> AbstractPipeline:
    """Instantiate a pipeline by ID."""
    cls = PIPELINE_REGISTRY.get(pipeline_id)
    if cls is None:
        raise ValueError(
            f"Unknown pipeline_id: {pipeline_id!r}. "
            f"Available: {list(PIPELINE_REGISTRY)}"
        )
    if artifacts_dir is not None:
        return cls(artifacts_dir=artifacts_dir)
    return cls()


__all__ = [
    "AbstractPipeline",
    "P0DeterministicBaseline",
    "P1WrongScope",
    "P2WrongFacts",
    "P3WrongAggregation",
    "P4CompoundSF",
    "P5FullCompound",
    "CounterfactualAttributor",
    "AttributionResult",
    "ComponentAttribution",
    "QueryFactory",
    "PIPELINE_REGISTRY",
    "get_pipeline",
]
