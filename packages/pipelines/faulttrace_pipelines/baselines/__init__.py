"""Abstract base class for baseline diagnostic adapters.

All baselines must implement this interface so that evaluation runs
against FaultTrace-RAG and all baselines are perfectly matched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosisInput:
    """A single case submitted to a diagnostic baseline."""

    case_id: str
    query: str
    retrieved_scope: list[dict[str, Any]]
    extracted_facts: list[dict[str, Any]]
    predicted_answer: Any
    gold_answer: Any
    error_magnitude: float
    model: str
    dataset: str


@dataclass
class DiagnosisOutput:
    """Output of a baseline diagnostic method for one case."""

    case_id: str
    predicted_faults: list[str]  # subset of {"R", "E", "A", "G", "COMPOUND"}
    confidence: float  # 0.0–1.0
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    raw_output: dict[str, Any] = field(default_factory=dict)


class BaselineAdapter(ABC):
    """Protocol that every diagnostic baseline must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this baseline."""

    @abstractmethod
    def diagnose(self, case: DiagnosisInput) -> DiagnosisOutput:
        """Run the diagnostic method on a single case and return fault labels."""

    def diagnose_batch(self, cases: list[DiagnosisInput]) -> list[DiagnosisOutput]:
        """Default sequential batch implementation; override for efficiency."""
        return [self.diagnose(c) for c in cases]
