from typing import Optional
from .base import DiagnosisPolicy
from .models import DiagnosticState

class GreedyPolicy(DiagnosisPolicy):
    def choose_next_intervention(self, state: DiagnosticState) -> Optional[int]:
        if not state.candidate_indices:
            return None
        for idx in state.candidate_indices:
            if idx not in state.observed:
                return idx
        return None
