import hashlib
import json
from typing import Any

from faulttrace_core.retrieval import RetrievalEngine, RetrievalUnit


class IndexManager:
    """Manages cached instances of retrieval engines to avoid rebuilding indexes."""

    def __init__(self):
        # Maps cache_key -> RetrievalEngine instance
        self._cache: dict[str, RetrievalEngine] = {}

    def _generate_key(
        self, units: list[RetrievalUnit], engine_type: str, config: dict[str, Any], dataset_id: str | None = None
    ) -> str:
        """Generate a stable hash based on corpus contents and index configuration."""
        # Text length is not a corpus fingerprint: different documents can have the
        # same IDs and lengths.  Include the ordered content hashes so cached engines
        # cannot silently cross datasets or corpus revisions.  Order is intentional
        # because it determines deterministic tie-breaking in the retrievers.
        corpus_signature = [
            (u.unit_id, u.record_id, u.content_hash())
            for u in units
        ]
        state = {
            "engine_type": engine_type,
            "config": config,
            "dataset_id": dataset_id,
            "corpus_signature": corpus_signature,
            "num_docs": len(units),
        }
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    def get_or_build(
        self,
        engine_type: str,
        units: list[RetrievalUnit],
        engine_instance: RetrievalEngine,
        config: dict[str, Any],
        dataset_id: str | None = None,
    ) -> RetrievalEngine:
        """
        Retrieves a cached engine if the corpus and config match exactly.
        Otherwise, builds the index on the provided instance and caches it.
        """
        cache_key = self._generate_key(units, engine_type, config, dataset_id=dataset_id)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Not found or invalidated, build the index
        engine_instance.build_index(units)
        self._cache[cache_key] = engine_instance
        return engine_instance

    def clear(self):
        """Clear all cached indexes."""
        self._cache.clear()


# Global index manager instance
index_manager = IndexManager()
