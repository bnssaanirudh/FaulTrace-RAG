from typing import Any

import torch
from sentence_transformers import SentenceTransformer, util

from faulttrace_core.retrieval import RetrievalEngine, RetrievalUnit


class DenseRetriever(RetrievalEngine):
    """Dense retrieval using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Use CPU explicitly if no GPU to avoid unexpected warnings on some envs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.units = []
        self.corpus_embeddings = None

    def build_index(self, units: list[RetrievalUnit], **kwargs):
        """Build the dense index from the given units."""
        self.units = units
        if not self.units:
            self.corpus_embeddings = None
            return

        texts = [unit.text for unit in self.units]
        # Encode all texts into a tensor matrix
        self.corpus_embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[dict[str, Any]]:
        if self.corpus_embeddings is None or len(self.units) == 0:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # Sort scores and indices
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.units)))

        results = []
        for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1], strict=False)):
            results.append(
                {"unit": self.units[idx.item()], "score": float(score.item()), "rank": rank + 1}
            )

        return results
