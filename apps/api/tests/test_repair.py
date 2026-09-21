import pytest
from typing import Dict, FrozenSet
from faulttrace_pipelines.repair.models import RepairResult
from faulttrace_pipelines.repair.mcr import minimal_repair_sets
from faulttrace_pipelines.repair.cost_aware import min_cost_repair
from faulttrace_pipelines.repair.objectives import weighted_objective_repair

def test_minimal_repair_sets():
    losses: Dict[FrozenSet[str], float] = {
        frozenset([]): 0.5,
        frozenset(["R"]): 0.3,
        frozenset(["E"]): 0.4,
        frozenset(["R", "E"]): 0.0,
        frozenset(["A"]): 0.0,
    }
    
    # Tolerance 0.1
    mcr = minimal_repair_sets(losses, ["R", "E", "A"], tolerance=0.1)
    # Expected min size is 1 (the set {"A"})
    assert mcr == [frozenset(["A"])]
    
    # Tie breaking for same size
    losses[frozenset(["E"])] = 0.0
    mcr = minimal_repair_sets(losses, ["R", "E", "A"], tolerance=0.1)
    assert mcr == [frozenset(["A"]), frozenset(["E"])]
    
    # Infeasible repair
    mcr = minimal_repair_sets(losses, ["R", "E", "A"], tolerance=-1.0)
    assert mcr == []

def test_min_cost_repair():
    losses: Dict[FrozenSet[str], float] = {
        frozenset([]): 0.5,
        frozenset(["R"]): 0.0,
        frozenset(["E"]): 0.0,
        frozenset(["R", "E"]): 0.0,
        frozenset(["A"]): 0.0,
    }
    
    costs = {"R": 10.0, "E": 2.0, "A": 5.0}
    
    # E is cheapest valid repair
    cmcr, cost = min_cost_repair(losses, ["R", "E", "A"], costs, tol=0.1)
    assert cmcr == frozenset(["E"])
    assert cost == 2.0

def test_weighted_objective_repair():
    losses: Dict[FrozenSet[str], float] = {
        frozenset([]): 0.5,
        frozenset(["R"]): 0.4,
        frozenset(["E"]): 0.1,
        frozenset(["R", "E"]): 0.0,
    }
    
    def cost_fn(S: FrozenSet[str]) -> float:
        return float(len(S)) * 0.15
    
    # L({}) + l*C({}) = 0.5 + 0 = 0.5
    # L({E}) + l*C({E}) = 0.1 + 0.15 = 0.25
    # L({R,E}) + l*C({R,E}) = 0.0 + 0.30 = 0.30
    best_S = weighted_objective_repair(losses, cost_fn, lambda_penalty=1.0)
    assert best_S == frozenset(["E"])
