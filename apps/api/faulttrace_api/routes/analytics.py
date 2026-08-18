from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from faulttrace_data.benchmarks.scifact import SciFactAdapter
from faulttrace_data.text.analytics import CorpusAnalytics
from pydantic import BaseModel

router = APIRouter()
DATA_ROOT = Path("data/benchmarks")


class AnalyticsResponse(BaseModel):
    summary: dict[str, Any]
    top_terms: list[dict[str, Any]]
    top_ngrams: list[dict[str, Any]]
    clusters: list[dict[str, Any]]


@router.get("/{dataset_id}", response_model=AnalyticsResponse)
def get_analytics(dataset_id: str):
    """Retrieve precomputed text-mining outputs for a given benchmark dataset."""
    docs = []
    if dataset_id == "scifact":
        adapter = SciFactAdapter(DATA_ROOT)
        try:
            docs = adapter.load_corpus()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="SciFact corpus not found")
    else:
        # Extend to covidqa, hotpotqa later or lazily if needed
        raise HTTPException(
            status_code=400, detail=f"Analytics not supported for dataset {dataset_id}"
        )

    analytics = CorpusAnalytics(docs)

    return AnalyticsResponse(
        summary=analytics.get_summary_stats(),
        top_terms=analytics.compute_tfidf_terms(top_n=20),
        top_ngrams=analytics.compute_ngrams(n=2, top_n=10),
        clusters=analytics.compute_clusters(n_clusters=5),
    )
