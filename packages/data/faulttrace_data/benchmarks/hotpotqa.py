from pathlib import Path

import pandas as pd
from faulttrace_core.retrieval import TextDocument


class HotpotQAAdapter:
    """Adapter for HotpotQA dataset."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.hotpot_dir = self.data_root / "hotpotqa" / "distractor"

    def load_corpus(self, splits: list[str] = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "validation-00000-of-00001.parquet"]) -> list[TextDocument]:
        """Extract unique documents from the context of all questions."""
        docs = {}
        for split in splits:
            file_path = self.hotpot_dir / split
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path, columns=["context"])
            for _, row in df.iterrows():
                context = row["context"]
                if not context or "title" not in context or "sentences" not in context:
                    continue

                titles = context["title"]
                sentences = context["sentences"]

                for title, sents in zip(titles, sentences):
                    # In HotpotQA, title serves as a unique document identifier (mostly)
                    if title not in docs:
                        text = " ".join(sents)
                        docs[title] = TextDocument(
                            doc_id=title,
                            title=title,
                            text=text,
                            metadata={"source": "hotpotqa"}
                        )
        return list(docs.values())

    def load_queries(self, split: str = "validation-00000-of-00001.parquet") -> dict[str, str]:
        """Load queries. Returns mapping of query_id -> text."""
        file_path = self.hotpot_dir / split
        if not file_path.exists():
            return {}

        df = pd.read_parquet(file_path, columns=["id", "question"])
        queries = {}
        for _, row in df.iterrows():
            queries[row["id"]] = row["question"]
        return queries

    def load_qrels(self, split: str = "validation-00000-of-00001.parquet") -> dict[str, dict[str, int]]:
        """Load supporting facts as qrels. Returns query_id -> {doc_id: 1}."""
        file_path = self.hotpot_dir / split
        if not file_path.exists():
            return {}

        df = pd.read_parquet(file_path, columns=["id", "supporting_facts"])
        qrels = {}
        for _, row in df.iterrows():
            q_id = row["id"]
            facts = row["supporting_facts"]
            qrels[q_id] = {}
            if facts and "title" in facts:
                for title in facts["title"]:
                    qrels[q_id][title] = 1
        return qrels
