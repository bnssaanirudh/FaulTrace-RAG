import math
from typing import Optional, List, Dict, Callable, Any
from .base import DiagnosisPolicy
from .models import DiagnosticState

class BayesianPolicy(DiagnosisPolicy):
    def __init__(self, template_mean: Dict[str, List[float]], cost_fn: Callable[[Any], float]):
        self.template_mean = template_mean
        self.cost_fn = cost_fn

    def choose_next_intervention(self, state: DiagnosticState) -> Optional[int]:
        choices = []
        for j in state.candidate_indices:
            if j in state.observed:
                continue

            means = [self.template_mean[h][j] for h in state.hypotheses]
            
            weighted_mean = sum(p * m for p, m in zip(state.posterior, means))
            disagreement = sum(p * ((m - weighted_mean) ** 2) for p, m in zip(state.posterior, means))
            
            S = state.intervention_sets[j]
            c = self.cost_fn(S)
            score = disagreement / max(c, 1e-9)
            
            choices.append((score, disagreement, -c, j))
            
        if not choices:
            return None
        return max(choices)[-1]
