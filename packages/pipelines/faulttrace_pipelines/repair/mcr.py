from typing import Dict, List, FrozenSet

def minimal_repair_sets(losses: Dict[FrozenSet[str], float], players: List[str], tolerance: float = 1e-9) -> List[FrozenSet[str]]:
    """All smallest intervention sets whose loss is at most tolerance."""
    valid = [S for S, loss in losses.items() if loss <= tolerance]
    if not valid:
        return []
    m = min(map(len, valid))
    return sorted([S for S in valid if len(S) == m], key=lambda s: tuple(sorted(s)))
