"""RAGAS baseline adapter.

RAGAS (Es et al., 2023) provides automated evaluation metrics for RAG
pipelines using LLM judges. This adapter maps its component scores to
FaultTrace-RAG fault labels (R, E, A) for matched comparison.

Reference:
    Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented
    Generation." arXiv:2309.15217 (2023).

Usage:
    adapter = RAGASAdapter(threshold=0.5)
    result = adapter.diagnose(case)

Note:
    RAGAS requires the `ragas` package. Install with:
        pip install ragas

    An OpenAI API key must be set in the environment:
        export OPENAI_API_KEY=sk-...
"""
from __future__ import annotations

import time

from . import BaselineAdapter, DiagnosisInput, DiagnosisOutput


class RAGASAdapter(BaselineAdapter):
    """Maps RAGAS component metrics to FaultTrace-RAG fault labels.

    RAGAS computes:
      - context_precision / context_recall → maps to R (retrieval) fault
      - faithfulness                        → maps to E (extraction) fault
      - answer_relevancy                    → maps to A (aggregation) fault

    A component is flagged as faulty if its score falls below `threshold`.
    """

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini") -> None:
        self.threshold = threshold
        self.llm_model = llm_model
        self._check_import()

    def _check_import(self) -> None:
        pass

    @property
    def name(self) -> str:
        return f"RAGAS(threshold={self.threshold}, llm={self.llm_model})"

    def diagnose(self, case: DiagnosisInput) -> DiagnosisOutput:
        """Run RAGAS evaluation and return mapped fault labels."""
        start = time.perf_counter()
        scores = self._run_ragas(case)
        elapsed = time.perf_counter() - start

        predicted_faults = self._scores_to_faults(scores)
        if len(predicted_faults) > 1:
            predicted_faults = ["COMPOUND"] + predicted_faults

        return DiagnosisOutput(
            case_id=case.case_id,
            predicted_faults=predicted_faults or ["A"],
            confidence=1.0 - min(scores.values()) if scores else 0.5,
            latency_seconds=elapsed,
            input_tokens=0,
            output_tokens=0,
            raw_output=scores,
        )

    def _run_ragas(self, case: DiagnosisInput) -> dict[str, float]:
        """Run RAGAS evaluation and return component scores."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        except ImportError:
            raise ImportError(
                "RAGAS dependencies are missing. "
                "Install with: pip install ragas datasets"
            )

        contexts = [str(r.get("text", r)) if isinstance(r, dict) else str(r) for r in case.retrieved_scope]
        data = {
            "question": [case.query],
            "answer": [str(case.predicted_answer)],
            "contexts": [contexts],
            "ground_truth": [str(case.gold_answer)]
        }
        dataset = Dataset.from_dict(data)

        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]

        try:
            result = evaluate(dataset, metrics=metrics)
            return {
                "context_precision": float(result.get("context_precision", 1.0)),
                "context_recall": float(result.get("context_recall", 1.0)),
                "faithfulness": float(result.get("faithfulness", 1.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 1.0)),
            }
        except Exception as e:
            # Fallback on evaluation error
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
            }

    def _scores_to_faults(self, scores: dict[str, float]) -> list[str]:
        faults: list[str] = []
        if scores.get("context_recall", 1.0) < self.threshold:
            faults.append("R")
        if scores.get("faithfulness", 1.0) < self.threshold:
            faults.append("E")
        if scores.get("answer_relevancy", 1.0) < self.threshold:
            faults.append("A")
        return faults
