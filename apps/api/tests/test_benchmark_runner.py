from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_index import index_manager
from faulttrace_pipelines.text_pipelines.benchmark_runner import BenchmarkRunner
from faulttrace_pipelines.text_pipelines.p_text_retrieve import PTextRetrieve


class MockPipeline:
    pipeline_id = "test_pipeline"
    def run(self, query_id, query_text, units, dataset_id, split, gold_support_status):
        if query_text == "fail_me":
            raise RuntimeError("Pipeline failed!")
        return {"query_id": query_id, "ranked_doc_ids": ["d1"]}

def test_benchmark_runner_robustness(tmp_path):
    runner = BenchmarkRunner(
        pipeline=MockPipeline(),
        dataset_id="test_ds",
        artifacts_dir=tmp_path / "artifacts"
    )

    queries = {
        "q1": "valid query",
        "q2": " ", # Empty query
        "q3": "fail_me", # Pipeline failure
    }

    qrels = {"q1": {"d1": 1}}

    res = runner.run_all(queries, units=[], qrels=qrels)

    assert res["total_queries"] == 3
    # Check that it didn't crash
    import pandas as pd
    df = pd.read_parquet(res["parquet_export"])

    assert len(df) == 3

    # Check empty query
    empty_res = df[df["query_id"] == "q2"].iloc[0]
    assert empty_res["status"] == "failed"
    assert empty_res["error"] == "Empty query text"

    # Check pipeline failure
    fail_res = df[df["query_id"] == "q3"].iloc[0]
    assert fail_res["status"] == "failed"
    assert "Pipeline failed!" in fail_res["error"]

    # Metrics should be computed for valid queries
    assert "retrieval_metrics" in res


def test_benchmark_runner_calls_real_retrieval_pipeline(tmp_path):
    """Exercise the production call contract rather than only a permissive mock."""
    index_manager.clear()
    units = [
        RetrievalUnit(unit_id="d1", record_id="d1", text="alpha evidence"),
        RetrievalUnit(unit_id="d2", record_id="d2", text="beta evidence"),
        RetrievalUnit(unit_id="d3", record_id="d3", text="gamma evidence"),
    ]
    runner = BenchmarkRunner(
        pipeline=PTextRetrieve(retriever_type="bm25", top_k=1),
        dataset_id="contract-fixture",
        artifacts_dir=tmp_path / "artifacts",
    )

    result = runner.run_all(
        {"q1": "alpha"},
        units,
        qrels={"q1": {"d1": 1}},
        split="test",
    )

    assert result["successful"] == 1
    assert result["failed"] == 0
    assert result["retrieval_metrics"]["recall@1"] == 1.0
