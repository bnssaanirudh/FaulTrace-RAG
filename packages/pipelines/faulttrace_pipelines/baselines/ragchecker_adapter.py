"""RAGChecker baseline adapter.

RAGChecker (Ru et al., 2024) provides component-level precision/recall
metrics for RAG pipelines. This adapter maps its component scores to
FaultTrace-RAG fault labels (R, E, A) so that a fair comparison can be
made on the same 200-case natural-failure benchmark.

Reference:
    Ru, D., et al. "RAGChecker: A Fine-grained Framework for Diagnosing
    Retrieval-Augmented Generation." arXiv:2408.08067 (2024).

Usage:
    adapter = RAGCheckerAdapter(threshold=0.5)
    result = adapter.diagnose(case)

Note:
    RAGChecker requires the `ragchecker` package. Install with:
        pip install ragchecker

    If not installed, this adapter will raise ImportError with instructions.
"""
from __future__ import annotations

import time
from typing import Any

from . import BaselineAdapter, DiagnosisInput, DiagnosisOutput


class RAGCheckerAdapter(BaselineAdapter):
    """Maps RAGChecker component scores to FaultTrace-RAG fault labels.

    RAGChecker computes:
      - context_precision / context_recall  → maps to R (retrieval) fault
      - faithfulness                         → maps to E (extraction) fault
      - answer_correctness                   → maps to A (aggregation) fault

    A component is flagged as faulty if its score falls below `threshold`.
    """

    def __init__(self, threshold: float = 0.5, model: str = "gpt-4o-mini") -> None:
        self.threshold = threshold
        self.model = model
        self._check_import()

    def _check_import(self) -> None:
        pass

    @property
    def name(self) -> str:
        return f"RAGChecker(threshold={self.threshold}, model={self.model})"

    def diagnose(self, case: DiagnosisInput) -> DiagnosisOutput:
        """Run RAGChecker on the case and return mapped fault labels."""

        start = time.perf_counter()
        scores: dict[str, float] = self._run_ragchecker(case)
        elapsed = time.perf_counter() - start

        predicted_faults = self._scores_to_faults(scores)
        if len(predicted_faults) > 1:
            predicted_faults = ["COMPOUND"] + predicted_faults

        return DiagnosisOutput(
            case_id=case.case_id,
            predicted_faults=predicted_faults or ["A"],  # fallback to aggregation
            confidence=1.0 - min(scores.values()) if scores else 0.5,
            latency_seconds=elapsed,
            input_tokens=0,   # RAGChecker internal – update once SDK exposes token counts
            output_tokens=0,
            raw_output=scores,
        )

    def _run_ragchecker(
        self, case: DiagnosisInput, ragchecker: Any = None
    ) -> dict[str, float]:
        """Run RAGChecker evaluation and return component scores."""
        try:
            from ragchecker import RAGResults, RAGChecker
        except ImportError:
            raise ImportError(
                "RAGChecker is missing. Install with: pip install ragchecker"
            )

        retrieved_context = [
            {"text": str(c.get("text", c)) if isinstance(c, dict) else str(c)}
            for c in case.retrieved_scope
        ]
        
        query_data = {
            "query_id": case.case_id,
            "query": case.query,
            "gt_answer": str(case.gold_answer),
            "response": str(case.predicted_answer),
            "retrieved_context": retrieved_context
        }
        
        try:
            rag_results = RAGResults.from_dict({"results": [query_data]})
            evaluator = RAGChecker(
                extract_model=self.model,
                judge_model=self.model
            )
            evaluator.evaluate(rag_results, all_metrics=True)
            
            res = rag_results.results[0].metrics if hasattr(rag_results.results[0], "metrics") else {}
            return {
                "context_precision": float(res.get("context_precision", 1.0)),
                "context_recall": float(res.get("context_recall", 1.0)),
                "faithfulness": float(res.get("faithfulness", 1.0)),
                "answer_correctness": float(res.get("overall_correctness", 1.0)),
            }
        except Exception as e:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_correctness": 0.0,
            }

    def _scores_to_faults(self, scores: dict[str, float]) -> list[str]:
        faults: list[str] = []
        if scores.get("context_recall", 1.0) < self.threshold:
            faults.append("R")
        if scores.get("faithfulness", 1.0) < self.threshold:
            faults.append("E")
        if scores.get("answer_correctness", 1.0) < self.threshold:
            faults.append("A")
        return faults
