import hashlib
import re
from pathlib import Path

import pandas as pd
from faulttrace_core.retrieval import TextDocument


class CovidQAAdapter:
    """Adapter for RAGBench COVID-QA dataset."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.covidqa_dir = self.data_root / "ragbench" / "covidqa"

    def load_corpus(self, splits: list[str] | None = None) -> list[TextDocument]:
        """Extract unique documents from the context of all questions."""
        if splits is None:
            splits = ["train-00000-of-00001.parquet", "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]
        docs = {}
        for split in splits:
            file_path = self.covidqa_dir / split
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path, columns=["id", "documents"])
            for _, row in df.iterrows():
                q_id = str(row["id"])
                documents = row["documents"]

                if documents is None:
                    continue

                for _idx, doc_text in enumerate(documents):
                    # We generate a deterministic doc ID based on text hash
                    doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:16]
                    if doc_hash not in docs:
                        docs[doc_hash] = TextDocument(
                            doc_id=doc_hash,
                            title=f"COVID-QA Document {doc_hash}",
                            text=doc_text,
                            metadata={"source": "covidqa", "original_query_id": q_id}
                        )
        return list(docs.values())

    def load_candidate_sets(
        self, split: str = "test-00000-of-00001.parquet"
    ) -> dict[str, list[TextDocument]]:
        """Load the four supplied context candidates for each RAGBench row."""
        file_path = self.covidqa_dir / split
        if not file_path.exists():
            return {}

        frame = pd.read_parquet(file_path, columns=["id", "documents"])
        candidate_sets: dict[str, list[TextDocument]] = {}
        for _, row in frame.iterrows():
            query_id = str(row["id"])
            documents = row["documents"]
            candidates: dict[str, TextDocument] = {}
            if documents is not None:
                for text in documents:
                    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    if doc_hash not in candidates:
                        candidates[doc_hash] = TextDocument(
                            doc_id=doc_hash,
                            title=f"COVID-QA Document {doc_hash}",
                            text=text,
                            metadata={"source": "covidqa", "query_id": query_id},
                        )
            candidate_sets[query_id] = list(candidates.values())
        return candidate_sets

    def load_queries(self, split: str = "test-00000-of-00001.parquet") -> dict[str, str]:
        """Load queries. Returns mapping of query_id -> text."""
        file_path = self.covidqa_dir / split
        if not file_path.exists():
            return {}

        df = pd.read_parquet(file_path, columns=["id", "question"])
        queries = {}
        for _, row in df.iterrows():
            queries[str(row["id"])] = row["question"]
        return queries

    def load_qrels(self, split: str = "test-00000-of-00001.parquet") -> dict[str, dict[str, int]]:
        """
        Load relevance labels. Returns query_id -> {doc_id: 1}.
        In COVID-QA, we map the provided documents and use relevance keys if available.
        For simplicity and consistency across splits, we treat any document that contains
        a relevant sentence key as a relevant document.
        """
        file_path = self.covidqa_dir / split
        if not file_path.exists():
            return {}

        # If 'all_relevant_sentence_keys' is not reliably present or easy to map without parsing,
        # we can use a simpler heuristic for the adapter if needed.
        # RAGBench documents list maps to characters (like '0' for first doc, '1' for second doc).
        # We will parse the relevant doc indices from 'all_relevant_sentence_keys'.
        df = pd.read_parquet(file_path)
        qrels = {}
        for _, row in df.iterrows():
            q_id = str(row["id"])
            documents = row.get("documents", [])
            rel_keys = row.get("all_relevant_sentence_keys", [])

            if documents is None or rel_keys is None or len(documents) == 0 or len(rel_keys) == 0:
                continue

            relevant: dict[str, int] = {}
            # Keys are encoded as ``<document index><sentence letter>``.  Parse
            # the complete numeric prefix rather than assuming a single digit.
            for key in rel_keys:
                match = re.match(r"^(\d+)", str(key))
                if match:
                    doc_idx = int(match.group(1))
                    if doc_idx < len(documents):
                        doc_text = documents[doc_idx]
                        doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:16]
                        relevant[doc_hash] = 1
            if relevant:
                qrels[str(q_id)] = relevant

        return qrels
