import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import faiss

from faulttrace_core.retrieval import RetrievalUnit, DocumentRenderer
from faulttrace_core.models import CorpusRecord

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns a 2D numpy array of shape (len(texts), dimension)."""
        pass
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def provider_id(self) -> str:
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding using SHA-256 (for tests)."""
    
    def __init__(self, dim: int = 16):
        self._dim = dim
        
    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Deterministic hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # Convert bytes to floats, pad/truncate to dim
            vec = [float(b) / 255.0 for b in h]
            if len(vec) < self._dim:
                vec.extend([0.0] * (self._dim - len(vec)))
            else:
                vec = vec[:self._dim]
                
            # Normalize for cosine similarity
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            embeddings.append(arr)
            
        return np.vstack(embeddings)
        
    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_id(self) -> str:
        return "mock_hash"


class SentenceTransformerProvider(EmbeddingProvider):
    """Real local embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required for this provider")
            
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
        
    def embed(self, texts: List[str]) -> np.ndarray:
        # Returns normalized embeddings suitable for inner-product -> cosine similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)
        
    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_id(self) -> str:
        return f"st_{self.model_name}"


class DenseRetriever:
    """FAISS-based dense retrieval for a CorpusWorld."""
    
    def __init__(
        self,
        world_id: str,
        data_dir: Path,
        renderer: DocumentRenderer,
        provider: EmbeddingProvider,
        batch_size: int = 32
    ):
        self.world_id = world_id
        self.data_dir = data_dir
        self.renderer = renderer
        self.provider = provider
        self.batch_size = batch_size
        
        self.units: List[RetrievalUnit] = []
        self._index: Optional[faiss.Index] = None
        
        self._load_and_index()
        
    def _load_and_index(self):
        """Loads records, embeds them, and builds the FAISS index."""
        world_dir = self.data_dir / "worlds" / self.world_id
        parquet_path = world_dir / "records.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found for world {self.world_id}")
            
        df = pd.read_parquet(parquet_path)
        
        # We use IndexFlatIP for Inner Product. If embeddings are normalized, IP == Cosine Similarity
        self._index = faiss.IndexFlatIP(self.provider.dimension)
        
        texts_to_embed = []
        for _, row in df.iterrows():
            record = CorpusRecord.model_validate(row.to_dict())
            units = self.renderer.render(record)
            for unit in units:
                self.units.append(unit)
                # Combine metadata heavily? For dense, we typically just embed the text.
                # Let's prepend some key metadata.
                meta_prefix = f"{unit.metadata.get('title', '')} {unit.metadata.get('category', '')}: "
                texts_to_embed.append(meta_prefix + unit.text)
                
        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[i:i + self.batch_size]
            emb = self.provider.embed(batch)
            all_embeddings.append(emb)
            
        if all_embeddings:
            final_embs = np.vstack(all_embeddings)
            self._index.add(final_embs)
            logger.info(f"Built FAISS index for {self.world_id} with {len(self.units)} units (dim={self.provider.dimension})")

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve top_k units using FAISS inner-product."""
        if not self._index or self._index.ntotal == 0:
            return []
            
        if not query:
            return []
            
        query_emb = self.provider.embed([query])
        
        # We need to retrieve more if we are going to filter locally.
        # Simple approach: over-fetch then filter.
        fetch_k = top_k * 10 if filters else top_k
        fetch_k = min(fetch_k, self._index.ntotal)
        
        scores, indices = self._index.search(query_emb, fetch_k)
        
        results = []
        rank = 1
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.units):
                continue
                
            unit = self.units[idx]
            
            if filters:
                match = True
                for k, v in filters.items():
                    if unit.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
                    
            results.append({
                "unit": unit,
                "score": float(score),
                "rank": rank
            })
            rank += 1
            
            if len(results) >= top_k:
                break
                
        return results
