import csv
import json
from pathlib import Path

from faulttrace_core.retrieval import TextDocument


class SciFactAdapter:
    """Adapter for BEIR SciFact dataset."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.beir_dir = self.data_root / "scifact" / "beir" / "scifact"

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
            header = next(reader)
            # expected header: query-id, corpus-id, score
            for row in reader:
                if len(row) < 3:
                    continue
                q_id, doc_id, score = row[0], row[1], int(row[2])
                if q_id not in qrels:
                    qrels[q_id] = {}
                qrels[q_id][doc_id] = score
        return qrels
