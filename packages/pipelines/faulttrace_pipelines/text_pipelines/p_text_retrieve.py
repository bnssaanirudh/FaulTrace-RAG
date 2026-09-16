"""
P-Text-Retrieve: Retrieval-only text benchmark pipeline.

Retrieves top-K documents for a query using the configured retriever.
Does NOT produce a CitedAnswer — only ranked document IDs and scores.
Useful as a baseline for measuring retrieval quality in isolation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_dense import DenseRetriever
from faulttrace_core.retrieval_hybrid import HybridRetriever
from faulttrace_core.retrieval_index import index_manager


class PTextRetrieve:
    """Retrieval-only pipeline for text benchmarks."""

    pipeline_id = "text-retrieve"

    def __init__(
        self,
        retriever_type: str = "hybrid",
        top_k: int = 10,
        artifacts_dir: Path = Path("artifacts/text_runs"),
    ):
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.artifacts_dir = artifacts_dir

    def run(
        self,
        query_id: str,
        query_text: str,
        units: list[RetrievalUnit],
        dataset_id: str = "unknown",
        split: str | None = None,
        gold_support_status: str | None = None,
    ) -> dict:
        """Run retrieval and return ranked results.

        ``split`` and ``gold_support_status`` are accepted as benchmark metadata so
        this concrete pipeline conforms to :class:`BenchmarkRunner`'s call contract.
        Gold labels are never consumed by retrieval.
        """
        t0 = time.perf_counter()

        bm25 = index_manager.get_or_build(
            "bm25", units, BM25Retriever(), {}, dataset_id=dataset_id
        )

        if self.retriever_type == "bm25":
            results = bm25.search(query_text, top_k=self.top_k)
        elif self.retriever_type == "dense":
            dense = index_manager.get_or_build(
                "dense", units, DenseRetriever(), {}, dataset_id=dataset_id
            )
            results = dense.search(query_text, top_k=self.top_k)
        else:  # hybrid
            dense = index_manager.get_or_build(
                "dense", units, DenseRetriever(), {}, dataset_id=dataset_id
            )
            hybrid = HybridRetriever(bm25, dense)
            hybrid.units = units
            results = hybrid.search(query_text, top_k=self.top_k)

        latency_ms = (time.perf_counter() - t0) * 1000

        ranked_ids = [r["unit"].record_id for r in results]
        ranked_scores = [r["score"] for r in results]

        return {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "split": split,
            "pipeline_id": self.pipeline_id,
            "retriever_type": self.retriever_type,
            "ranked_doc_ids": ranked_ids,
            "ranked_scores": ranked_scores,
            "top_k": self.top_k,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now(UTC).isoformat(),
        }
