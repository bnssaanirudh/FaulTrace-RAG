from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
import hashlib

from faulttrace_core.retrieval import RetrievalUnit

class ContextManifest(BaseModel):
    """Manifest describing the built context for tracing."""
    candidate_count: int
    selected_count: int
    dropped_count: int
    total_estimated_tokens: int
    truncation_reason: str = "none"
    context_hash: str
    selected_unit_ids: List[str]

class ContextBuilder:
    """Packs retrieval units into a context string within a token budget."""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        
    def build_context(self, candidates: List[RetrievalUnit]) -> Tuple[str, ContextManifest]:
        """
        Builds a single string context from candidates, dropping those that exceed the token budget.
        Returns the formatted context string and the tracking manifest.
        """
        selected_units = []
        dropped_count = 0
        current_tokens = 0
        truncation_reason = "none"
        
        for unit in candidates:
            if current_tokens + unit.token_estimate <= self.max_tokens:
                selected_units.append(unit)
                current_tokens += unit.token_estimate
            else:
                dropped_count += 1
                if truncation_reason == "none":
                    truncation_reason = "token_budget_exceeded"
                    
        # Format the context
        context_parts = []
        for i, unit in enumerate(selected_units):
            context_parts.append(f"--- Document [{i+1}] (ID: {unit.record_id}) ---\n{unit.text}")
            
        full_context = "\n\n".join(context_parts)
        
        # Calculate stable hash of the context
        context_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest() if selected_units else ""
        # Fix: using full_context
        context_hash = hashlib.sha256(full_context.encode("utf-8")).hexdigest() if selected_units else ""

        manifest = ContextManifest(
            candidate_count=len(candidates),
            selected_count=len(selected_units),
            dropped_count=dropped_count,
            total_estimated_tokens=current_tokens,
            truncation_reason=truncation_reason,
            context_hash=context_hash,
            selected_unit_ids=[u.unit_id for u in selected_units]
        )
        
        return full_context, manifest
