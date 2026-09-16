from pathlib import Path

import pandas as pd
from faulttrace_core.retrieval import TextDocument


class HotpotQAAdapter:
    """Adapter for HotpotQA dataset."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.hotpot_dir = self.data_root / "hotpotqa" / "distractor"

    def load_corpus(self, splits: list[str] | None = None) -> list[TextDocument]:
        """Extract unique documents from the context of all questions."""
        if splits is None:
            splits = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "validation-00000-of-00001.parquet"]
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

                for title, sents in zip(titles, sentences, strict=False):
                    text = " ".join(sents)
                    # A small number of titles have more than one text variant in
                    # the Parquet conversion. Preserve all unique variants under
                    # the title-level qrel identifier instead of silently retaining
                    # whichever row happened to occur first.
                    if title in docs and text not in docs[title].text:
                        docs[title].text += f"\n{text}"
                    elif title not in docs:
                        docs[title] = TextDocument(
                            doc_id=title,
                            title=title,
                            text=text,
                            metadata={"source": "hotpotqa"}
                        )
        return list(docs.values())

    def load_candidate_sets(
        self, split: str = "validation-00000-of-00001.parquet"
    ) -> dict[str, list[TextDocument]]:
        """Load the official per-question distractor candidate sets."""
        file_path = self.hotpot_dir / split
        if not file_path.exists():
            return {}

        frame = pd.read_parquet(file_path, columns=["id", "context"])
        candidate_sets: dict[str, list[TextDocument]] = {}
        for _, row in frame.iterrows():
            by_title: dict[str, TextDocument] = {}
            context = row["context"]
            for title, sentences in zip(
                context["title"], context["sentences"], strict=False
            ):
                title = str(title)
                text = " ".join(map(str, sentences))
                if title in by_title and text not in by_title[title].text:
                    by_title[title].text += f"\n{text}"
                elif title not in by_title:
                    by_title[title] = TextDocument(
                        doc_id=title,
                        title=title,
                        text=text,
                        metadata={"source": "hotpotqa", "query_id": str(row["id"])},
                    )
            candidate_sets[str(row["id"])] = list(by_title.values())
        return candidate_sets

    def load_queries(self, split: str = "validation-00000-of-00001.parquet") -> dict[str, str]:
        """Load queries. Returns mapping of query_id -> text."""
        file_path = self.hotpot_dir / split
        if not file_path.exists():
            return {}

        df = pd.read_parquet(file_path, columns=["id", "question"])
        queries = {}
        for _, row in df.iterrows():
            queries[str(row["id"])] = row["question"]
        return queries

    def load_qrels(self, split: str = "validation-00000-of-00001.parquet") -> dict[str, dict[str, int]]:
        """Load supporting facts as qrels. Returns query_id -> {doc_id: 1}."""
        file_path = self.hotpot_dir / split
        if not file_path.exists():
            return {}

        df = pd.read_parquet(file_path, columns=["id", "supporting_facts"])
        qrels = {}
        for _, row in df.iterrows():
            q_id = str(row["id"])
            facts = row["supporting_facts"]
            qrels[q_id] = {}
            if facts and "title" in facts:
                for title in facts["title"]:
                    qrels[q_id][title] = 1
        return qrels
