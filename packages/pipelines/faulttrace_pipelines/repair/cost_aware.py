from typing import Dict, List, FrozenSet, Tuple, Optional

def min_cost_repair(losses: Dict[FrozenSet[str], float], players: List[str], costs: Dict[str, float], tol: float) -> Tuple[Optional[FrozenSet[str]], float]:
    """Finds the valid intervention set with the minimum cost."""
    valid = [S for S, l in losses.items() if l <= tol]
    if not valid:
        return None, float("inf")
    
    def cost_fn(S: FrozenSet[str]) -> float:
        return sum(costs.get(p, 1.0) for p in S)
    
    best_S = min(valid, key=lambda S: (cost_fn(S), len(S), tuple(sorted(S))))
    return best_S, cost_fn(best_S)
