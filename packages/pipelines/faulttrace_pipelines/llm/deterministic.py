import hashlib
import json
import logging
from typing import Any, Dict, Optional

from faulttrace_core.llm import ModelProvider, ProviderConfig, ModelOutput, register_provider

logger = logging.getLogger(__name__)

class DeterministicProvider(ModelProvider):
    """
    Test provider that returns outputs deterministically based on the prompt hash.
    Can inject errors if specific model_ids are requested.
    """

    @classmethod
    def provider_id(cls) -> str:
        return "deterministic"

    def generate(self, prompt: str, config: ProviderConfig) -> ModelOutput:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        
        # Token estimation is deterministic
        prompt_tokens = self.estimate_tokens(prompt)
        
        raw_response = f"Deterministic response for {prompt_hash}"
        parsed_json: Optional[Dict[str, Any]] = None
        rejected_reason: Optional[str] = None
        completion_tokens = 10
        
        if config.structured_schema:
            if "malformed" in config.model_id:
                raw_response = "{ this is not valid json ]"
                rejected_reason = "json_parse_error"
            elif "omission" in config.model_id:
                raw_response = "{}"
                parsed_json = {}
                # The pipeline schema validator will reject it since required fields are missing
            elif "arithmetic_error" in config.model_id:
                # Generate valid JSON but maybe with intentionally bad numbers for downstream tasks to catch
                parsed_json = self._generate_dummy_json(config.structured_schema)
                # If there's a field expecting numbers, mess it up
                _corrupt_numbers(parsed_json)
                raw_response = json.dumps(parsed_json)
            else:
                parsed_json = self._generate_dummy_json(config.structured_schema)
                raw_response = json.dumps(parsed_json)
        else:
            if "wrong" in config.model_id:
                raw_response = "INCORRECT_ANSWER"
            elif "insufficient" in config.model_id:
                raw_response = "INSUFFICIENT_EVIDENCE"
            
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
        """Rough estimation: 1 token per 4 chars."""
        return max(1, len(text) // 4)

    def _generate_dummy_json(self, schema: Dict[str, Any]) -> Any:
        """Generates a dummy object based on a simple JSON schema."""
        schema_type = schema.get("type", "object")
        if schema_type == "object":
            obj = {}
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                obj[key] = self._generate_dummy_json(prop_schema)
            return obj
        elif schema_type == "array":
            items_schema = schema.get("items", {})
            return [self._generate_dummy_json(items_schema)]
        elif schema_type == "string":
            if "enum" in schema:
                return schema["enum"][0]
            return "dummy_string"
        elif schema_type in ("integer", "number"):
            return 42
        elif schema_type == "boolean":
            return True
        elif schema_type == "null":
            return None
        else:
            return "unknown_type"

def _corrupt_numbers(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                obj[k] = v + 1000
            else:
                _corrupt_numbers(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], (int, float)) and not isinstance(obj[i], bool):
                obj[i] = obj[i] + 1000
            else:
                _corrupt_numbers(obj[i])

# Auto-register
register_provider(DeterministicProvider)
