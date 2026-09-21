from dataclasses import dataclass
from typing import FrozenSet

@dataclass
class RepairResult:
    interventions: FrozenSet[str]
    cost: float
    residual_loss: float
