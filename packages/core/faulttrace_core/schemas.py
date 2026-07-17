"""
JSON Schema export utilities for all public FaultTrace-RAG contracts.

Exports schemas as versioned JSON files for documentation and API validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faulttrace_core.models import (
    AggregationSpec,
    ComponentOutput,
    CorpusRecord,
    CorpusWorld,
    CoverageCertificate,
    FactSpec,
    GoldAnswer,
    PipelineRun,
    QuerySpec,
    TraceEvent,
    SCHEMA_VERSION,
)


EXPORTABLE_MODELS = {
    "CorpusRecord": CorpusRecord,
    "CorpusWorld": CorpusWorld,
    "FactSpec": FactSpec,
    "GoldAnswer": GoldAnswer,
    "PipelineRun": PipelineRun,
    "QuerySpec": QuerySpec,
    "TraceEvent": TraceEvent,
    "ComponentOutput": ComponentOutput,
    "CoverageCertificate": CoverageCertificate,
}


def export_all_schemas(output_dir: Path) -> dict[str, Path]:
    """Export JSON schemas for all public contracts to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, model_class in EXPORTABLE_MODELS.items():
        schema = model_class.model_json_schema()  # type: ignore[attr-defined]
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["x-schema-version"] = SCHEMA_VERSION
        out_path = output_dir / f"{name}.schema.json"
        out_path.write_text(json.dumps(schema, indent=2))
        written[name] = out_path
    return written


def get_schema(model_name: str) -> dict[str, Any]:
    """Get JSON schema dict for a named model."""
    if model_name not in EXPORTABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(EXPORTABLE_MODELS)}")
    return EXPORTABLE_MODELS[model_name].model_json_schema()  # type: ignore[attr-defined]
