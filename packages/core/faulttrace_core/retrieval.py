import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from faulttrace_core.models import CorpusRecord

class RetrievalUnit(BaseModel):
    """A unit of text retrieved from the corpus, representing a whole record or a chunk."""
    unit_id: str = Field(..., description="Unique ID for this specific chunk/unit")
    record_id: str = Field(..., description="ID of the source CorpusRecord")
    text: str = Field(..., description="The actual text content to be embedded or passed to the LLM")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Fields available for filtering/structured access")
    chunk_index: int = Field(default=0, description="Index of the chunk within the record, 0 if unchunked")
    token_estimate: int = Field(default=0, description="Estimated token count")
    
    def content_hash(self) -> str:
        """Stable hash of the text content."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

class DocumentRenderer:
    """Deterministically renders a CorpusRecord into text strings and chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def render(self, record: CorpusRecord) -> List[RetrievalUnit]:
        """Render a CorpusRecord into one or more RetrievalUnits."""
        # Simple rendering template
        lines = [
            f"Title: {record.title}",
            f"Brand: {record.brand}",
            f"Category: {record.category.value}",
            f"Price: {record.price if record.price is not None else 'N/A'}",
            f"Rating: {record.rating}",
            f"Date: {record.event_time.strftime('%Y-%m-%d')}",
            f"Verified Purchase: {record.verified_purchase}",
            f"Review: {record.text}"
        ]
        full_text = "\n".join(lines)
        metadata = {
            "title": record.title,
            "brand": record.brand,
            "category": record.category.value,
            "rating": record.rating,
            "verified_purchase": record.verified_purchase,
            "event_time": record.event_time.isoformat()
        }

        # Token estimation heuristic (1 token per 4 chars)
        def est_tokens(t: str) -> int:
            return max(1, len(t) // 4)

        if not self.chunk_size:
            return [RetrievalUnit(
                unit_id=f"{record.record_id}_0",
                record_id=record.record_id,
                text=full_text,
                metadata=metadata,
                chunk_index=0,
                token_estimate=est_tokens(full_text)
            )]

        # Chunking (character-based chunking for simplicity; ideally word/token based)
        chunks = []
        text_len = len(full_text)
        start = 0
        idx = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = full_text[start:end]
            chunks.append(RetrievalUnit(
                unit_id=f"{record.record_id}_{idx}",
                record_id=record.record_id,
                text=chunk_text,
                metadata=metadata,
                chunk_index=idx,
                token_estimate=est_tokens(chunk_text)
            ))
            idx += 1
            if end == text_len:
                break
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks
