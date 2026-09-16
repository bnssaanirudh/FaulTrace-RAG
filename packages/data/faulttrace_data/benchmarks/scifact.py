import csv
import json
from dataclasses import dataclass
from pathlib import Path

from faulttrace_core.retrieval import TextDocument


@dataclass(frozen=True)
class SciFactClaimCase:
    query_id: str
    claim: str
    gold_support_status: str
    evidence_doc_statuses: dict[str, str]
    cited_doc_ids: tuple[str, ...]


class SciFactAdapter:
    """Adapter for BEIR SciFact dataset."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.beir_dir = self.data_root / "scifact" / "beir" / "scifact"
        self.official_dir = self.data_root / "scifact" / "official" / "data"

    def load_corpus(self) -> list[TextDocument]:
        """Load corpus.jsonl into TextDocuments."""
        corpus_file = self.beir_dir / "corpus.jsonl"
        if not corpus_file.exists():
            raise FileNotFoundError(f"SciFact corpus not found at {corpus_file}")

        docs = []
        with open(corpus_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                docs.append(TextDocument(
                    doc_id=str(record.get("_id", record.get("id"))),
                    title=record.get("title", ""),
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {})
                ))
        return docs

    def load_queries(self) -> dict[str, str]:
        """Load queries.jsonl. Returns mapping of query_id -> text."""
        queries_file = self.beir_dir / "queries.jsonl"
        if not queries_file.exists():
            raise FileNotFoundError(f"SciFact queries not found at {queries_file}")

        queries = {}
        with open(queries_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                q_id = str(record.get("_id", record.get("id")))
                queries[q_id] = record.get("text", "")
        return queries

    def load_qrels(self, split: str = "test") -> dict[str, dict[str, int]]:
        """Load qrels from {split}.tsv. Returns mapping of query_id -> {doc_id: score}"""
        qrels_file = self.beir_dir / "qrels" / f"{split}.tsv"
        if not qrels_file.exists():
            raise FileNotFoundError(f"SciFact qrels not found at {qrels_file}")

        qrels = {}
        with open(qrels_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            # expected header: query-id, corpus-id, score
            for row in reader:
                if len(row) < 3:
                    continue
                q_id, doc_id, score = row[0], row[1], int(row[2])
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][doc_id] = score
        return qrels

    def load_official_corpus(self) -> list[TextDocument]:
        """Load the original SciFact corpus used by the labeled claim splits."""
        corpus_file = self.official_dir / "corpus.jsonl"
        if not corpus_file.exists():
            raise FileNotFoundError(f"Official SciFact corpus not found at {corpus_file}")
        documents = []
        with corpus_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                documents.append(
                    TextDocument(
                        doc_id=str(record["doc_id"]),
                        title=record.get("title", ""),
                        text=" ".join(record.get("abstract", [])),
                        metadata={
                            "source": "scifact-official",
                            "structured": bool(record.get("structured", False)),
                        },
                    )
                )
        return documents

    def load_official_claims(self, split: str = "dev") -> list[SciFactClaimCase]:
        """Load labeled original-format claims without exposing labels to the pipeline."""
        claims_file = self.official_dir / f"claims_{split}.jsonl"
        if not claims_file.exists():
            raise FileNotFoundError(f"Official SciFact claims not found at {claims_file}")
        cases = []
        with claims_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                evidence_statuses: dict[str, str] = {}
                for doc_id, evidence_sets in record.get("evidence", {}).items():
                    labels = {item["label"] for item in evidence_sets}
                    if len(labels) != 1:
                        raise ValueError(
                            f"Claim {record['id']} has conflicting labels for document {doc_id}"
                        )
                    label = labels.pop()
                    evidence_statuses[str(doc_id)] = {
                        "SUPPORT": "supported",
                        "CONTRADICT": "unsupported",
                    }[label]
                claim_labels = set(evidence_statuses.values())
                if not claim_labels:
                    gold_status = "insufficient_evidence"
                elif len(claim_labels) == 1:
                    gold_status = claim_labels.pop()
                else:
                    gold_status = "conflicting"
                cases.append(
                    SciFactClaimCase(
                        query_id=str(record["id"]),
                        claim=record["claim"],
                        gold_support_status=gold_status,
                        evidence_doc_statuses=evidence_statuses,
                        cited_doc_ids=tuple(str(value) for value in record.get("cited_doc_ids", [])),
                    )
                )
        return cases
