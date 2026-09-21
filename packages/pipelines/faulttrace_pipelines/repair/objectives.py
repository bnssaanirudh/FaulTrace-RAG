from typing import Dict, FrozenSet, Callable

def weighted_objective_repair(losses: Dict[FrozenSet[str], float], cost_fn: Callable[[FrozenSet[str]], float], lambda_penalty: float) -> FrozenSet[str]:
    """Minimizes L(S) + lambda * C(S)."""
    return min(losses.keys(), key=lambda S: losses[S] + lambda_penalty * cost_fn(S))
