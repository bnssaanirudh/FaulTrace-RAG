import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from faulttrace_core.models import _stable_hash, _utcnow
from faulttrace_core.llm import ModelOutput

class CacheEntry(BaseModel):
    cache_key: str
    provider_name: str
    model_id: str
    schema_version: str
    is_success: bool
    parsed_json: Optional[Dict[str, Any]] = None
    raw_response: str = ""
    rejected_reason: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    software_version: str = "1.0.0"

class ExtractionCache:
    """
    WP4: Content-Addressed Extraction Cache.
    A disk-based cache for extraction units. Keyed by the fingerprint of model,
    prompt, schema, records, and repair attempt.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Default to a global cache dir so resuming across runs works
            self.cache_dir = Path("data/cache/extraction")
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_key(
        self,
        provider_name: str,
        model_id: str,
        prompt_hash: str,
        records_hash: str,
        schema_hash: str,
        repair_attempt: int
    ) -> str:
        data = {
            "provider": provider_name,
            "model": model_id,
            "prompt_hash": prompt_hash,
            "records_hash": records_hash,
            "schema_hash": schema_hash,
            "repair_attempt": repair_attempt
        }
        return _stable_hash(data)

    def _get_path(self, cache_key: str) -> Path:
        # Split into subdirs to avoid huge flat directories
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.json"

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        path = self._get_path(cache_key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return CacheEntry.model_validate(data)
            except Exception:
                return None
        return None

    def put(self, entry: CacheEntry) -> None:
        path = self._get_path(entry.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(entry.model_dump_json())

    def store_output(
        self,
        cache_key: str,
        output: ModelOutput,
        schema_version: str,
        latency_ms: float
    ) -> CacheEntry:
        is_success = output.parsed_json is not None
        
        entry = CacheEntry(
            cache_key=cache_key,
            provider_name=output.provider_name or "unknown",
            model_id=output.model_id or "unknown",
            schema_version=schema_version,
            is_success=is_success,
            parsed_json=output.parsed_json,
            raw_response=output.raw_response,
            rejected_reason=output.rejected_reason,
            latency_ms=latency_ms,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens
        )
        self.put(entry)
        return entry
