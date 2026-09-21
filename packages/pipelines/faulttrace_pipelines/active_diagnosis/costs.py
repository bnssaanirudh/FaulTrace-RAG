from typing import Any, Callable, Dict, FrozenSet

def default_intervention_cost(S: FrozenSet[str], stage_cost: Dict[str, float] = None) -> float:
    """Calculates cost of an intervention set."""
    if stage_cost is None:
        return 0.25 + len(S)
    return 0.25 + sum(stage_cost.get(p, 1.0) for p in S)
