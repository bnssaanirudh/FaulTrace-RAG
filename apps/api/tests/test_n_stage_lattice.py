import math
import pytest
from faulttrace_core.models import (
    CounterfactualWorld,
    InterventionSet,
    PipelineGraph,
    PipelineRun,
    PipelineStage,
    RunStatus,
)
from faulttrace_pipelines.n_stage_lattice import GeneralLatticeRunner


def create_dummy_run():
    return PipelineRun(
        run_id="dummy-run",
        query_id="query",
        pipeline_id="dummy-pipeline",
        provider_id="deterministic",
        status="completed",
        answer=10,
        gold_answer_value=20,
        is_correct=False,
        is_within_tolerance=False,
        loss=0.5,
        latency_ms=100.0,
        token_estimate_input=0,
        token_estimate_output=0,
        artifact_references={},
    )


def test_n1_lattice():
    runner = GeneralLatticeRunner()
    run = create_dummy_run()
    graph = PipelineGraph(
        stages={"S1": PipelineStage(stage_id="S1")},
        edges=[]
    )

    # v(none) = 0, v(S1) = 0.5 (loss drops by 0.5)
    def evaluate(subset: set[str]):
        loss = 0.5 if not subset else 0.0
        # For simplicity, returning an object that quacks like LossDiagnostic
        class DummyDiag:
            normalized_loss = loss
        return CounterfactualWorld(
            intervention=InterventionSet(replaced_stages=subset),
            status=RunStatus.COMPLETED,
            loss_diagnostic=DummyDiag()
        )

    summary = runner.execute_lattice(run, graph, {}, 0.5, evaluate)
    assert len(summary.worlds) == 2
    assert summary.shapley_values["S1"] == 0.5
    assert summary.interaction == 0.0


def test_n2_lattice():
    runner = GeneralLatticeRunner()
    run = create_dummy_run()
    graph = PipelineGraph(
        stages={
            "A": PipelineStage(stage_id="A"),
            "B": PipelineStage(stage_id="B")
        },
        edges=[("A", "B")]
    )

    # Values: v(none)=0, v(A)=0.2, v(B)=0.3, v(A,B)=0.8
    # phi_A = 0.5 * (0.2 - 0) + 0.5 * (0.8 - 0.3) = 0.1 + 0.25 = 0.35
    # phi_B = 0.5 * (0.3 - 0) + 0.5 * (0.8 - 0.2) = 0.15 + 0.3 = 0.45
    # interaction = 0.8 - 0.8 = 0
    def evaluate(subset: set[str]):
        if not subset: loss = 0.8
        elif subset == {"A"}: loss = 0.6
        elif subset == {"B"}: loss = 0.5
        else: loss = 0.0

        class DummyDiag:
            normalized_loss = loss
        return CounterfactualWorld(
            intervention=InterventionSet(replaced_stages=subset),
            status=RunStatus.COMPLETED,
            loss_diagnostic=DummyDiag()
        )

    summary = runner.execute_lattice(run, graph, {}, 0.8, evaluate)
    assert len(summary.worlds) == 4
    assert math.isclose(summary.shapley_values["A"], 0.35)
    assert math.isclose(summary.shapley_values["B"], 0.45)


def test_n3_lattice_compat():
    runner = GeneralLatticeRunner()
    run = create_dummy_run()
    graph = PipelineGraph(
        stages={
            "R": PipelineStage(stage_id="R"),
            "E": PipelineStage(stage_id="E"),
            "A": PipelineStage(stage_id="A")
        },
        edges=[("R", "E"), ("E", "A")]
    )

    def evaluate(subset: set[str]):
        loss = 1.0 - 0.1 * len(subset)
        class DummyDiag:
            normalized_loss = loss
        return CounterfactualWorld(
            intervention=InterventionSet(replaced_stages=subset),
            status=RunStatus.COMPLETED,
            loss_diagnostic=DummyDiag()
        )

    summary = runner.execute_lattice(run, graph, {}, 1.0, evaluate)
    assert len(summary.worlds) == 8


def test_n5_lattice():
    runner = GeneralLatticeRunner()
    run = create_dummy_run()
    graph = PipelineGraph(
        stages={
            "S": PipelineStage(stage_id="S"),
            "R": PipelineStage(stage_id="R"),
            "E": PipelineStage(stage_id="E"),
            "A": PipelineStage(stage_id="A"),
            "G": PipelineStage(stage_id="G"),
        },
        edges=[("S", "R"), ("R", "E"), ("E", "A"), ("A", "G")]
    )

    count = 0
    def evaluate(subset: set[str]):
        nonlocal count
        count += 1
        class DummyDiag:
            normalized_loss = 0.0
        return CounterfactualWorld(
            intervention=InterventionSet(replaced_stages=subset),
            status=RunStatus.COMPLETED,
            loss_diagnostic=DummyDiag()
        )

    summary = runner.execute_lattice(run, graph, {}, 0.5, evaluate)
    assert len(summary.worlds) == 32
    assert count == 32


def test_invalid_dag():
    with pytest.raises(ValueError, match="cycle"):
        PipelineGraph(
            stages={
                "A": PipelineStage(stage_id="A"),
                "B": PipelineStage(stage_id="B")
            },
            edges=[("A", "B"), ("B", "A")]
        )


def test_complete_lattice_enumeration_missing_oracle_failure():
    runner = GeneralLatticeRunner()
    run = create_dummy_run()
    graph = PipelineGraph(
        stages={"A": PipelineStage(stage_id="A")},
        edges=[]
    )

    def evaluate(subset: set[str]):
        if subset:
            # Simulate failure when oracle is applied
            return CounterfactualWorld(
                intervention=InterventionSet(replaced_stages=subset),
                status=RunStatus.FAILED,
                natural_language_summary="missing oracle definition"
            )
        class DummyDiag:
            normalized_loss = 0.5
        return CounterfactualWorld(
            intervention=InterventionSet(replaced_stages=subset),
            status=RunStatus.COMPLETED,
            loss_diagnostic=DummyDiag()
        )

    with pytest.raises(ValueError, match="missing oracle"):
        runner.execute_lattice(run, graph, {}, 0.5, evaluate)
