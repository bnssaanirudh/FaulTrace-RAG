from typing import Optional
from .models import DiagnosticState

class DiagnosisPolicy:
    def choose_next_intervention(self, state: DiagnosticState) -> Optional[int]:
        raise NotImplementedError
