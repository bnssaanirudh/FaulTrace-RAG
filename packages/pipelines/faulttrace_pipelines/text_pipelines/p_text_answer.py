"""
P-Text-Answer: Retrieval + evidence-grounded direct answer pipeline.

Retrieves top-K documents and produces a CitedAnswer with:
- support_status
- cited_doc_ids
- supporting_quote
- abstention_reason (if applicable)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faulttrace_core.evidence import CitedAnswer, SupportStatus
from faulttrace_core.extraction_providers import (
    BoundedRepairExtractor,
    DeterministicFixtureExtractor,
)
from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_dense import DenseRetriever
from faulttrace_core.retrieval_hybrid import HybridRetriever
from faulttrace_core.retrieval_index import index_manager


class PTextAnswer:
    """Retrieval + evidence-grounded answer pipeline."""

    pipeline_id = "text-answer"

    def __init__(
        self,
        retriever_type: str = "hybrid",
        top_k: int = 5,
        max_repair_retries: int = 3,
        artifacts_dir: Path = Path("artifacts/text_runs"),
    ):
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.artifacts_dir = artifacts_dir
        base_extractor = DeterministicFixtureExtractor()
        self.extractor = BoundedRepairExtractor(base_extractor, max_retries=max_repair_retries)

    def run(
        self,
        query_id: str,
        query_text: str,
        units: list[RetrievalUnit],
        dataset_id: str = "unknown",
        split: str = "test",
        gold_support_status: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve docs then produce a cited answer."""
        t0 = time.perf_counter()

        # Stage 1: Retrieve
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

        retrieve_latency_ms = (time.perf_counter() - t0) * 1000

        # Stage 2: Extract from each retrieved doc
        t1 = time.perf_counter()
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

        extract_latency_ms = (time.perf_counter() - t1) * 1000

        # Stage 3: Aggregate support status across documents
        t2 = time.perf_counter()
        final_answer = self._aggregate_answers(query_id, extraction_records)
        aggregate_latency_ms = (time.perf_counter() - t2) * 1000

        total_latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "pipeline_id": self.pipeline_id,
            "support_status": final_answer.support_status.value,
            "answer_text": final_answer.answer_text,
            "cited_doc_ids": final_answer.cited_doc_ids,
            "cited_chunk_ids": final_answer.cited_chunk_ids,
            "supporting_quote": final_answer.supporting_quote,
            "abstention_reason": final_answer.abstention_reason,
            "gold_support_status": gold_support_status,
            "is_correct": (
                final_answer.support_status.value == gold_support_status
                if gold_support_status
                else None
            ),
            "docs_retrieved": len(retrieved),
            "docs_extracted": len(extraction_records),
            "repair_log": [a.to_dict() for a in self.extractor.repair_log],
            "latency_ms": {
                "retrieve": round(retrieve_latency_ms, 2),
                "extract": round(extract_latency_ms, 2),
                "aggregate": round(aggregate_latency_ms, 2),
                "total": round(total_latency_ms, 2),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _aggregate_answers(self, query_id: str, records) -> CitedAnswer:
        """Aggregate support status from multiple extraction records."""
        if not records:
            return CitedAnswer(
                support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
                abstention_reason="No documents retrieved",
            )

        # Collect all statuses
        statuses = []
        all_cited_ids = []
        best_quote = None

        for rec in records:
            if rec.cited_answer:
                statuses.append(rec.cited_answer.support_status)
                all_cited_ids.extend(rec.cited_answer.cited_doc_ids)
                if (
                    best_quote is None
                    and rec.cited_answer.supporting_quote
                    and rec.cited_answer.support_status == SupportStatus.SUPPORTED
                ):
                    best_quote = rec.cited_answer.supporting_quote

        if not statuses:
            return CitedAnswer(
                support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
                cited_doc_ids=[],
                abstention_reason="No extraction outputs available",
            )

        # Aggregation logic
        status_set = set(statuses)
        if SupportStatus.CONFLICTING in status_set:
            final_status = SupportStatus.CONFLICTING
            abstention = "Conflicting evidence found across retrieved documents"
        elif SupportStatus.SUPPORTED in status_set and SupportStatus.UNSUPPORTED in status_set:
            final_status = SupportStatus.CONFLICTING
            abstention = "Some documents support while others refute the claim"
        elif SupportStatus.SUPPORTED in status_set:
            count = statuses.count(SupportStatus.SUPPORTED)
            final_status = (
                SupportStatus.SUPPORTED if count >= 1 else SupportStatus.PARTIALLY_SUPPORTED
            )
            abstention = None
        elif SupportStatus.PARTIALLY_SUPPORTED in status_set:
            final_status = SupportStatus.PARTIALLY_SUPPORTED
            abstention = None
        elif SupportStatus.UNSUPPORTED in status_set:
            final_status = SupportStatus.UNSUPPORTED
            abstention = None
        else:
            final_status = SupportStatus.INSUFFICIENT_EVIDENCE
            abstention = "Evidence was retrieved but none supported or refuted the claim"

        return CitedAnswer(
            answer_text=best_quote,
            support_status=final_status,
            cited_doc_ids=list(dict.fromkeys(all_cited_ids)),  # deduplicate, preserve order
            supporting_quote=best_quote,
            abstention_reason=abstention,
        )
