import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
import pandas as pd

from faulttrace_core.retrieval import RetrievalUnit, DocumentRenderer
from faulttrace_core.models import CorpusRecord

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Deterministic tokenization and normalization."""
    if not text:
        return []
    # Lowercase and split by non-alphanumeric
    tokens = re.split(r'[^a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]

class BM25Retriever:
    """BM25 retrieval for a CorpusWorld."""
    
    def __init__(
        self,
        world_id: str,
        data_dir: Path,
        renderer: DocumentRenderer,
        metadata_weight: float = 2.0
    ):
        self.world_id = world_id
        self.data_dir = data_dir
        self.renderer = renderer
        self.metadata_weight = metadata_weight
        
        self.units: List[RetrievalUnit] = []
        self._bm25: Optional[BM25Okapi] = None
        
        self._load_and_index()
        
    def _load_and_index(self):
        """Loads records from the world's parquet file and builds the BM25 index."""
        world_dir = self.data_dir / "worlds" / self.world_id
        parquet_path = world_dir / "records.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found for world {self.world_id}")
            
        df = pd.read_parquet(parquet_path)
        
        tokenized_corpus = []
        for _, row in df.iterrows():
            record = CorpusRecord.model_validate(row.to_dict())
            units = self.renderer.render(record)
            for unit in units:
                self.units.append(unit)
                # Combine text and metadata for search, overweighting metadata if requested
                metadata_str = f"{unit.metadata.get('title', '')} {unit.metadata.get('brand', '')} {unit.metadata.get('category', '')}"
                
                # Apply metadata weighting by simply repeating metadata tokens
                meta_tokens = tokenize(metadata_str)
                text_tokens = tokenize(unit.text)
                
                weighted_tokens = text_tokens + (meta_tokens * int(self.metadata_weight))
                tokenized_corpus.append(weighted_tokens)
                
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index for {self.world_id} with {len(self.units)} units")

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top_k units.
        Returns a list of dicts with 'unit', 'score', and 'rank'.
        Filters are exact match on unit.metadata.
        """
        if not self._bm25:
            return []
            
        query_tokens = tokenize(query)
        if not query_tokens:
            # Handle empty queries by returning initial units
            scores = [0.0] * len(self.units)
        else:
            scores = self._bm25.get_scores(query_tokens)
            
        # Pair with indices
        scored_indices = list(enumerate(scores))
        
        # Sort by score descending
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        rank = 1
        
        for idx, score in scored_indices:
            unit = self.units[idx]
            
            # Apply filters if present
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
