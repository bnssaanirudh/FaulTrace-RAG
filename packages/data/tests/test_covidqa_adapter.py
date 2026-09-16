import hashlib

import pandas as pd
from faulttrace_data.benchmarks.covidqa import CovidQAAdapter


def test_covidqa_qrels_parse_full_document_index_and_drop_unlabeled(tmp_path):
    benchmark_dir = tmp_path / "ragbench" / "covidqa"
    benchmark_dir.mkdir(parents=True)
    documents = [f"document {index}" for index in range(11)]
    frame = pd.DataFrame(
        [
            {
                "id": "labeled",
                "documents": documents,
                "all_relevant_sentence_keys": ["10a", "10b"],
            },
            {
                "id": "unlabeled",
                "documents": documents,
                "all_relevant_sentence_keys": [],
            },
        ]
    )
    filename = "test-00000-of-00001.parquet"
    frame.to_parquet(benchmark_dir / filename, index=False)

    qrels = CovidQAAdapter(tmp_path).load_qrels(filename)
    expected = hashlib.sha256(documents[10].encode("utf-8")).hexdigest()[:16]

    assert qrels == {"labeled": {expected: 1}}


def test_covidqa_candidate_sets_deduplicate_identical_document_text(tmp_path):
    benchmark_dir = tmp_path / "ragbench" / "covidqa"
    benchmark_dir.mkdir(parents=True)
    filename = "test-00000-of-00001.parquet"
    pd.DataFrame(
        [{"id": "q1", "documents": ["same", "same", "different"]}]
    ).to_parquet(benchmark_dir / filename, index=False)

    candidates = CovidQAAdapter(tmp_path).load_candidate_sets(filename)

    assert len(candidates["q1"]) == 2
    assert len({candidate.doc_id for candidate in candidates["q1"]}) == 2
