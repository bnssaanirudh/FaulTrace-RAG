"""
BenchmarkRunner: Orchestrates any text pipeline over a full dataset's query set.

Persists results as Parquet + JSON and produces evaluation metrics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from faulttrace_core.evaluation import evaluate_retrieval
from faulttrace_core.retrieval import RetrievalUnit


class BenchmarkRunner:
    """
    Runs a text pipeline over a full dataset and persists results.

    Compatible with PTextRetrieve, PTextAnswer, and PTextExtract.
    """

    def __init__(
        self,
        pipeline,
        dataset_id: str,
        artifacts_dir: Path = Path("artifacts/benchmark_runs"),
    ):
        self.pipeline = pipeline
        self.dataset_id = dataset_id
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        queries: dict[str, str],
        units: list[RetrievalUnit],
        qrels: dict[str, dict[str, int]] | None = None,
        split: str = "test",
        max_queries: int | None = None,
    ) -> dict[str, Any]:
        """
        Run pipeline over all queries and collect results.

        Args:
            queries: dict mapping query_id -> query_text
            units: pre-built retrieval units for the corpus
            qrels: optional dict mapping query_id -> {doc_id: relevance}
            split: dataset split name
            max_queries: limit number of queries (for testing)
        """
        run_id = f"{self.dataset_id}_{self.pipeline.pipeline_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        results = []

        query_ids = list(queries.keys())
        if max_queries:
            query_ids = query_ids[:max_queries]

        for qid in query_ids:
            query_text = queries[qid]
            gold_status = None

            try:
                result = self.pipeline.run(
                    query_id=qid,
                    query_text=query_text,
                    units=units,
                    dataset_id=self.dataset_id,
                    split=split,
                    gold_support_status=gold_status,
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "query_id": qid,
                        "dataset_id": self.dataset_id,
                        "pipeline_id": self.pipeline.pipeline_id,
                        "error": str(e),
                        "status": "failed",
                    }
                )

        # Compute retrieval metrics if qrels provided
        retrieval_metrics = {}
        if qrels and hasattr(results[0], "get") and results[0].get("ranked_doc_ids"):
            ranked_results = {
                r["query_id"]: r["ranked_doc_ids"] for r in results if "ranked_doc_ids" in r
            }
            retrieval_metrics = evaluate_retrieval(ranked_results, qrels)

        # Persist
        df = pd.DataFrame(results)
        parquet_path = self.artifacts_dir / f"{run_id}.parquet"
        json_path = self.artifacts_dir / f"{run_id}.json"

        df.to_parquet(parquet_path, index=False)

        summary = {
            "run_id": run_id,
            "dataset_id": self.dataset_id,
            "pipeline_id": self.pipeline.pipeline_id,
            "split": split,
            "total_queries": len(query_ids),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "retrieval_metrics": retrieval_metrics,
            "parquet_export": str(parquet_path),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary
