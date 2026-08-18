from pathlib import Path

import pytest
from faulttrace_core.evaluation import evaluate_retrieval
from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_data.benchmarks.scifact import SciFactAdapter

DATA_ROOT = Path("data/benchmarks")


def test_scifact_adapter_loads_data():
    adapter = SciFactAdapter(DATA_ROOT)

    # We may not have the data fully present during a generic test run without downloading it,
    # but let's test if the paths are correctly resolved
    try:
        docs = adapter.load_corpus()
        assert len(docs) > 0

        queries = adapter.load_queries()
        assert len(queries) > 0

        qrels = adapter.load_qrels()
        assert len(qrels) > 0

    except FileNotFoundError:
        pytest.skip("SciFact dataset not found locally, skipping data load test")


def test_bm25_retriever_logic():
    retriever = BM25Retriever()
    units = [
        RetrievalUnit(
            unit_id="doc1", record_id="doc1", text="The quick brown fox jumps over the lazy dog"
        ),
        RetrievalUnit(
            unit_id="doc2", record_id="doc2", text="Machine learning is fascinating and complex"
        ),
        RetrievalUnit(unit_id="doc3", record_id="doc3", text="The lazy dog sleeps all day"),
    ]

    retriever.build_index(units)

    results = retriever.search("lazy dog", top_k=2)
    assert len(results) == 2

    # "The lazy dog sleeps all day" is shorter and has both words, should be highly ranked
    # "The quick brown fox jumps over the lazy dog" also has both words
    top_doc_ids = [r["unit"].record_id for r in results]
    assert "doc1" in top_doc_ids
    assert "doc3" in top_doc_ids
    assert "doc2" not in top_doc_ids


def test_evaluation_metrics():
    results = {"q1": ["doc1", "doc2", "doc3"], "q2": ["doc3", "doc1", "doc2"]}

    qrels = {"q1": {"doc1": 1, "doc4": 1}, "q2": {"doc2": 1}}

    metrics = evaluate_retrieval(results, qrels, k_values=[1, 3])

    # q1: doc1 is rank 1 (hit). MRR = 1.0. Recall@3 = 1/2. Precision@3 = 1/3
    # q2: doc2 is rank 3 (hit). MRR = 1/3. Recall@3 = 1/1. Precision@3 = 1/3

    # Average MRR = (1.0 + 0.333) / 2 = 0.666
    assert abs(metrics["mrr"] - 0.6666) < 0.01

    # Average Recall@3 = (0.5 + 1.0) / 2 = 0.75
    assert abs(metrics["recall@3"] - 0.75) < 0.01

    # Average Precision@3 = (0.333 + 0.333) / 2 = 0.333
    assert abs(metrics["precision@3"] - 0.3333) < 0.01
