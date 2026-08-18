from typing import Any

from faulttrace_core.retrieval import RetrievalEngine, RetrievalUnit


class HybridRetriever(RetrievalEngine):
    """Hybrid retrieval combining two other RetrievalEngines using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, engine_a: RetrievalEngine, engine_b: RetrievalEngine, rrf_k: int = 60):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.rrf_k = rrf_k
        self.units = []

    def build_index(self, units: list[RetrievalUnit], **kwargs):
        """Build indexes for both underlying engines."""
        self.units = units
        self.engine_a.build_index(units, **kwargs)
        self.engine_b.build_index(units, **kwargs)

    def search(self, query: str, top_k: int = 10, **kwargs) -> list[dict[str, Any]]:
        if not self.units:
            return []

        # Fetch more candidates to ensure good fusion
        fetch_k = max(top_k * 2, 50)

        results_a = self.engine_a.search(query, top_k=fetch_k, **kwargs)
        results_b = self.engine_b.search(query, top_k=fetch_k, **kwargs)

        rrf_scores = {}
        unit_map = {}

        # Calculate RRF for Engine A
        for item in results_a:
            u_id = item["unit"].unit_id
            unit_map[u_id] = item["unit"]
            rank = item["rank"]
            rrf_scores[u_id] = rrf_scores.get(u_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Calculate RRF for Engine B
        for item in results_b:
            u_id = item["unit"].unit_id
            unit_map[u_id] = item["unit"]
            rank = item["rank"]
            rrf_scores[u_id] = rrf_scores.get(u_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort combined scores
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (u_id, score) in enumerate(sorted_rrf[:top_k]):
            results.append({"unit": unit_map[u_id], "score": float(score), "rank": rank + 1})

        return results
