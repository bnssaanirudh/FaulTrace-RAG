import json

from faulttrace_data.benchmarks.scifact import SciFactAdapter


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_official_scifact_adapter_maps_claim_labels_and_corpus(tmp_path):
    official = tmp_path / "scifact" / "official" / "data"
    official.mkdir(parents=True)
    _write_jsonl(
        official / "corpus.jsonl",
        [
            {
                "doc_id": 10,
                "title": "Study",
                "abstract": ["Sentence one.", "Sentence two."],
                "structured": False,
            }
        ],
    )
    _write_jsonl(
        official / "claims_dev.jsonl",
        [
            {
                "id": 1,
                "claim": "A supported claim",
                "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [10],
            },
            {
                "id": 2,
                "claim": "A contradicted claim",
                "evidence": {"10": [{"label": "CONTRADICT", "sentences": [1]}]},
                "cited_doc_ids": [10],
            },
            {"id": 3, "claim": "Unknown", "evidence": {}, "cited_doc_ids": [10]},
        ],
    )

    adapter = SciFactAdapter(tmp_path)
    documents = adapter.load_official_corpus()
    cases = adapter.load_official_claims("dev")

    assert documents[0].doc_id == "10"
    assert documents[0].text == "Sentence one. Sentence two."
    assert [case.gold_support_status for case in cases] == [
        "supported",
        "unsupported",
        "insufficient_evidence",
    ]
    assert cases[0].evidence_doc_statuses == {"10": "supported"}
