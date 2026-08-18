import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from faulttrace_core.evaluation import evaluate_retrieval
from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_dense import DenseRetriever
from faulttrace_core.retrieval_hybrid import HybridRetriever
from faulttrace_core.retrieval_index import index_manager
from faulttrace_data.benchmarks.scifact import SciFactAdapter
from pydantic import BaseModel

router = APIRouter()
DATA_ROOT = Path("data/benchmarks")


class RetrievalCompareRequest(BaseModel):
    query: str
    dataset_id: str
    top_k: int = 10


class RetrievalMetricsResponse(BaseModel):
    bm25_metrics: dict[str, float]
    dense_metrics: dict[str, float]
    hybrid_metrics: dict[str, float]
    queries_evaluated: int


def _load_dataset_units(dataset_id: str) -> list[RetrievalUnit]:
    if dataset_id == "scifact":
        adapter = SciFactAdapter(DATA_ROOT)
        docs = adapter.load_corpus()
    else:
        raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} not supported yet.")

    # Convert TextDocuments to RetrievalUnits
    units = []
    for doc in docs:
        units.append(
            RetrievalUnit(
                unit_id=doc.doc_id,
                record_id=doc.doc_id,
                text=f"{doc.title} {doc.text}",  # Simple title + text concatenation for indexing
                metadata=doc.metadata,
                chunk_index=0,
            )
        )
    return units


@router.post("/compare")
def compare_retrievers(req: RetrievalCompareRequest):
    """Compare BM25, Dense, and Hybrid retrievers on a specific query."""
    units = _load_dataset_units(req.dataset_id)

    # Initialize engines
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    # Get or build from cache
    bm25 = index_manager.get_or_build("bm25", units, bm25, {})
    dense = index_manager.get_or_build("dense", units, dense, {})

    hybrid = HybridRetriever(bm25, dense)
    hybrid.units = (
        units  # Hybrid doesn't need its own build_index cache if underlying engines are cached
    )

    t0 = time.perf_counter()
    bm25_res = bm25.search(req.query, top_k=req.top_k)
    t1 = time.perf_counter()

    dense_res = dense.search(req.query, top_k=req.top_k)
    t2 = time.perf_counter()

    hybrid_res = hybrid.search(req.query, top_k=req.top_k)
    t3 = time.perf_counter()

    # Format results
    def format_res(results):
        return [
            {
                "doc_id": r["unit"].record_id,
                "score": r["score"],
                "text": r["unit"].text[:200] + "...",
            }
            for r in results
        ]

    return {
        "bm25": {"latency_ms": (t1 - t0) * 1000, "results": format_res(bm25_res)},
        "dense": {"latency_ms": (t2 - t1) * 1000, "results": format_res(dense_res)},
        "hybrid": {"latency_ms": (t3 - t2) * 1000, "results": format_res(hybrid_res)},
    }


@router.get("/metrics")
def get_metrics(
    dataset_id: str = "scifact", limit: int = Query(50, description="Max queries to evaluate")
):
    """Evaluate retrieval pipelines against the dataset's qrels."""
    units = _load_dataset_units(dataset_id)

    if dataset_id != "scifact":
        raise HTTPException(
            status_code=400, detail="Only scifact evaluation is implemented for now."
        )

    adapter = SciFactAdapter(DATA_ROOT)
    queries = adapter.load_queries()
    qrels = adapter.load_qrels()

    # Filter queries that have qrels
    eval_queries = {q_id: q_text for q_id, q_text in queries.items() if q_id in qrels}

    # Limit number of queries to avoid long runtimes during testing
    q_ids = list(eval_queries.keys())[:limit]

    bm25 = index_manager.get_or_build("bm25", units, BM25Retriever(), {})
    dense = index_manager.get_or_build("dense", units, DenseRetriever(), {})
    hybrid = HybridRetriever(bm25, dense)
    hybrid.units = units

    bm25_results = {}
    dense_results = {}
    hybrid_results = {}

    for q_id in q_ids:
        q_text = eval_queries[q_id]

        b_res = bm25.search(q_text, top_k=10)
        bm25_results[q_id] = [r["unit"].record_id for r in b_res]

        d_res = dense.search(q_text, top_k=10)
        dense_results[q_id] = [r["unit"].record_id for r in d_res]

        h_res = hybrid.search(q_text, top_k=10)
        hybrid_results[q_id] = [r["unit"].record_id for r in h_res]

    return RetrievalMetricsResponse(
        bm25_metrics=evaluate_retrieval(bm25_results, qrels),
        dense_metrics=evaluate_retrieval(dense_results, qrels),
        hybrid_metrics=evaluate_retrieval(hybrid_results, qrels),
        queries_evaluated=len(q_ids),
    )
