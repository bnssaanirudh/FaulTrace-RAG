import string
from typing import Any

from rank_bm25 import BM25Okapi

from faulttrace_core.retrieval import RetrievalEngine, RetrievalUnit


class BM25Retriever(RetrievalEngine):
    """Lexical retrieval using BM25Okapi."""

    def __init__(self):
        self.bm25 = None
        self.units = []

    def _tokenize(self, text: str) -> list[str]:
        # Simple whitespace tokenization and punctuation removal for BM25
        translator = str.maketrans("", "", string.punctuation)
        return text.lower().translate(translator).split()

    def build_index(self, units: list[RetrievalUnit], **kwargs):
        """Build the BM25 index from the given units."""
        self.units = units
        tokenized_corpus = [self._tokenize(unit.text) for unit in self.units]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[dict[str, Any]]:
        if not self.bm25 or not self.units:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort scores descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            # Only return items with a non-zero score (or let them all pass if needed, but 0 means no match)
            if scores[idx] > 0:
                results.append(
                    {"unit": self.units[idx], "score": float(scores[idx]), "rank": rank + 1}
                )
        return results
