import json
import logging
from typing import Any, Dict, Optional

import tiktoken
from openai import OpenAI, OpenAIError

from faulttrace_core.llm import ModelProvider, ProviderConfig, ModelOutput, register_provider

logger = logging.getLogger(__name__)

class OpenAIProvider(ModelProvider):
    """
    OpenAI-compatible provider. Can be used for OpenAI API, local Ollama, vLLM, etc.
    Requires OPENAI_API_KEY environment variable if hitting OpenAI.
    """

    @classmethod
    def provider_id(cls) -> str:
        return "openai"

    def generate(self, prompt: str, config: ProviderConfig) -> ModelOutput:
        client_kwargs: Dict[str, Any] = {}
        if config.endpoint_url:
            client_kwargs["base_url"] = config.endpoint_url
            # Ollama requires an API key, even if empty or dummy
            client_kwargs["api_key"] = "dummy"

        try:
            client = OpenAI(**client_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return ModelOutput(
                raw_response="",
                rejected_reason=f"client_init_error: {str(e)}",
                provider_name=self.provider_id(),
                model_id=config.model_id
            )

        messages = [
            {"role": "user", "content": prompt}
        ]

        kwargs: Dict[str, Any] = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout_seconds,
        }

        if config.structured_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.structured_schema_name or "output",
                    "schema": config.structured_schema,
                    "strict": True
                }
            }

        prompt_tokens = 0
        completion_tokens = 0
        raw_response = ""
        parsed_json: Optional[Dict[str, Any]] = None
        rejected_reason: Optional[str] = None

        try:
            response = client.chat.completions.create(**kwargs)
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            choice = response.choices[0]
            raw_response = choice.message.content or ""
            
            if config.structured_schema:
                try:
                    parsed_json = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    rejected_reason = f"json_parse_error: {str(e)}"
        
        except OpenAIError as e:
            logger.error(f"OpenAI generation error: {e}")
            rejected_reason = f"api_error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}")
            rejected_reason = f"internal_error: {str(e)}"

        return ModelOutput(
            raw_response=raw_response,
            parsed_json=parsed_json,
            rejected_reason=rejected_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider_name=self.provider_id(),
            model_id=config.model_id
        )

    def estimate_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

# Auto-register
register_provider(OpenAIProvider)
