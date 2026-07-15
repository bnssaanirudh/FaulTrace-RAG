from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from faulttrace_core.llm import _PROVIDERS, ProviderConfig, get_provider

router = APIRouter()

@router.get("/providers", summary="List active model providers")
async def list_providers() -> List[Dict[str, Any]]:
    """Return all registered model providers."""
    providers = []
    for pid, pcls in _PROVIDERS.items():
        providers.append({
            "provider_id": pid,
            "class": pcls.__name__
        })
    return providers

@router.get("/providers/{provider_id}/test", summary="Test provider connectivity")
async def test_provider(provider_id: str, model_id: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Test connectivity to a specific provider."""
    try:
        provider_cls = get_provider(provider_id)
        provider = provider_cls()
        config = ProviderConfig(model_id=model_id, max_tokens=10)
        
        # Test with a simple prompt
        result = provider.generate("Say 'hello'", config)
        return {
            "status": "success",
            "provider_id": provider_id,
            "model_id": model_id,
            "response": result.raw_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipelines", summary="List available pipelines")
async def list_pipelines() -> List[str]:
    """Return all available pipeline IDs."""
    from faulttrace_pipelines import PIPELINE_REGISTRY
    return list(PIPELINE_REGISTRY.keys())
