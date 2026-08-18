"""
P-Text-Extract: Retrieval + structured extraction + deterministic aggregation.

Full three-stage pipeline:
  R: Retrieve top-K docs using hybrid retrieval
  E: Extract facts and build ExtractionRecords per doc
  A: Aggregate into AggregatedExtractionResult + ProvenanceGraph
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faulttrace_core.evidence import AggregatedExtractionResult
from faulttrace_core.extraction_providers import (
    BoundedRepairExtractor,
    DeterministicFixtureExtractor,
)
from faulttrace_core.knowledge_graph import ProvenanceGraphBuilder
from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_dense import DenseRetriever
from faulttrace_core.retrieval_hybrid import HybridRetriever
from faulttrace_core.retrieval_index import index_manager

from faulttrace_pipelines.text_pipelines.p_text_answer import PTextAnswer


class PTextExtract:
    """Full structured extraction pipeline with provenance graph output."""

    pipeline_id = "text-extract"

    def __init__(
        self,
        retriever_type: str = "hybrid",
        top_k: int = 5,
        max_repair_retries: int = 3,
        artifacts_dir: Path = Path("artifacts/text_runs"),
        save_graph: bool = True,
    ):
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.artifacts_dir = artifacts_dir
        self.save_graph = save_graph
        base_extractor = DeterministicFixtureExtractor()
        self.extractor = BoundedRepairExtractor(base_extractor, max_retries=max_repair_retries)
        self._answer_pipeline = PTextAnswer(
            retriever_type=retriever_type, top_k=top_k, max_repair_retries=max_repair_retries
        )

    def run(
        self,
        query_id: str,
        query_text: str,
        units: list[RetrievalUnit],
        dataset_id: str = "unknown",
        split: str = "test",
        gold_support_status: str | None = None,
    ) -> dict[str, Any]:
        """Run full extraction pipeline and build provenance graph."""
        t0 = time.perf_counter()

        # Stage 1 + 2: Retrieval + per-doc extraction (reuse PTextAnswer logic)
        bm25 = index_manager.get_or_build("bm25", units, BM25Retriever(), {})
        if self.retriever_type == "bm25":
            retrieved = bm25.search(query_text, top_k=self.top_k)
        elif self.retriever_type == "dense":
            dense = index_manager.get_or_build("dense", units, DenseRetriever(), {})
            retrieved = dense.search(query_text, top_k=self.top_k)
        else:
            dense = index_manager.get_or_build("dense", units, DenseRetriever(), {})
            hybrid = HybridRetriever(bm25, dense)
            hybrid.units = units
            retrieved = hybrid.search(query_text, top_k=self.top_k)

        extraction_records = []
        for item in retrieved:
            unit = item["unit"]
            record = self.extractor.extract(
                doc_id=unit.record_id,
                doc_text=unit.text,
                query=query_text,
                dataset_id=dataset_id,
                split=split,
            )
            record = record.model_copy(update={"query_id": query_id})
            extraction_records.append(record)

        # Stage 3: Aggregate into AggregatedExtractionResult
        final_answer = self._answer_pipeline._aggregate_answers(query_id, extraction_records)

        # Build aggregated result (validates multi-doc citation integrity)
        try:
            agg_result = AggregatedExtractionResult(
                query_id=query_id,
                dataset_id=dataset_id,
                split=split,
                records=extraction_records,
                final_answer=final_answer,
                provider=self.extractor.provider_name,
            )
            citation_integrity_ok = True
            citation_integrity_error = None
        except Exception as e:
            agg_result = None
            citation_integrity_ok = False
            citation_integrity_error = str(e)

        # Stage 4: Build provenance graph
        graph_path = None
        graph_stats = {}
        if self.save_graph and agg_result:
            try:
                builder = ProvenanceGraphBuilder(
                    graph_id=f"{dataset_id}_{query_id}",
                    dataset_id=dataset_id,
                )
                builder.add_dataset_node(dataset_id, f"Dataset: {dataset_id}")
                builder.ingest_aggregated_result(agg_result)
                graph = builder.build()
                graph_stats = graph.stats()

                self.artifacts_dir.mkdir(parents=True, exist_ok=True)
                graph_path = self.artifacts_dir / f"graph_{query_id}.json"
                graph.save(graph_path)
            except Exception as e:
                graph_stats = {"error": str(e)}

        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "pipeline_id": self.pipeline_id,
            "support_status": final_answer.support_status.value,
            "answer_text": final_answer.answer_text,
            "cited_doc_ids": final_answer.cited_doc_ids,
            "supporting_quote": final_answer.supporting_quote,
            "abstention_reason": final_answer.abstention_reason,
            "gold_support_status": gold_support_status,
            "is_correct": (
                final_answer.support_status.value == gold_support_status
                if gold_support_status
                else None
            ),
            "citation_integrity_ok": citation_integrity_ok,
            "citation_integrity_error": citation_integrity_error,
            "total_facts_extracted": sum(len(r.facts) for r in extraction_records),
            "docs_retrieved": len(retrieved),
            "docs_extracted": len(extraction_records),
            "graph_stats": graph_stats,
            "graph_path": str(graph_path) if graph_path else None,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        return result
