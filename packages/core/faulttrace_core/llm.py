from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Type
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    """Configuration for a specific model provider execution."""
    model_id: str = Field(..., description="Provider-specific model ID (e.g., 'gpt-4o' or 'llama3')")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    endpoint_url: Optional[str] = Field(None, description="Custom endpoint for OpenAI-compatible services")
    structured_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for structured output")
    structured_schema_name: Optional[str] = Field(None, description="Name of the structured schema")
    timeout_seconds: float = Field(default=60.0)
    max_retries: int = Field(default=1)

class ModelOutput(BaseModel):
    """Result of a model generation request."""
    raw_response: str
    parsed_json: Optional[Dict[str, Any]] = None
    rejected_reason: Optional[str] = Field(None, description="Reason if parsing/validation failed")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider_name: str
    model_id: str

    def is_successful_structured(self) -> bool:
        return self.parsed_json is not None and self.rejected_reason is None

class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    def generate(self, prompt: str, config: ProviderConfig) -> ModelOutput:
        """Execute a generation request."""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens for the given text."""
        pass

    @classmethod
    @abstractmethod
    def provider_id(cls) -> str:
        """Return the unique identifier for this provider."""
        pass

# Provider Registry
_PROVIDERS: Dict[str, Type[ModelProvider]] = {}

def register_provider(provider_cls: Type[ModelProvider]) -> None:
    _PROVIDERS[provider_cls.provider_id()] = provider_cls

def get_provider(provider_id: str) -> Type[ModelProvider]:
    if provider_id not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_id}' not found. Registered: {list(_PROVIDERS.keys())}")
    return _PROVIDERS[provider_id]
