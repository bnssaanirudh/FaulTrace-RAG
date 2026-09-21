from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DiagnosticState:
    candidate_indices: List[int]
    hypotheses: List[str]
    posterior: List[float]
    observed: Dict[int, float]
    intervention_sets: List[Any]
    intervention_names: List[str]

@dataclass
class DiagnosticResult:
    pred: str
    n_probes: int
    confidence: float
    observed_names: List[str]
