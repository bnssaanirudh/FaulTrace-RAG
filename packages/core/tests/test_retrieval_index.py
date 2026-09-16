from faulttrace_core.retrieval import RetrievalUnit
from faulttrace_core.retrieval_bm25 import BM25Retriever
from faulttrace_core.retrieval_index import IndexManager


def _unit(text: str) -> RetrievalUnit:
    return RetrievalUnit(unit_id="same-id", record_id="same-record", text=text)


def test_cache_fingerprint_includes_content_not_only_length():
    manager = IndexManager()
    first = manager.get_or_build("bm25", [_unit("alpha")], BM25Retriever(), {})
    second = manager.get_or_build("bm25", [_unit("bravo")], BM25Retriever(), {})

    assert first is not second
    assert first.units[0].text == "alpha"
    assert second.units[0].text == "bravo"


def test_cache_key_separates_dataset_identities():
    manager = IndexManager()
    units = [_unit("alpha")]
    first = manager.get_or_build(
        "bm25", units, BM25Retriever(), {}, dataset_id="dataset-a"
    )
    second = manager.get_or_build(
        "bm25", units, BM25Retriever(), {}, dataset_id="dataset-b"
    )

    assert first is not second
