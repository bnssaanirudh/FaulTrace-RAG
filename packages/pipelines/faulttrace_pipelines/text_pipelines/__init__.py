"""Text-benchmark pipeline registry (separate from PIPELINE_REGISTRY for Track M)."""

from faulttrace_pipelines.text_pipelines.benchmark_runner import BenchmarkRunner
from faulttrace_pipelines.text_pipelines.p_text_answer import PTextAnswer
from faulttrace_pipelines.text_pipelines.p_text_extract import PTextExtract
from faulttrace_pipelines.text_pipelines.p_text_retrieve import PTextRetrieve

TEXT_PIPELINE_REGISTRY = {
    "text-retrieve": PTextRetrieve,
    "text-answer": PTextAnswer,
    "text-extract": PTextExtract,
}

__all__ = [
    "PTextRetrieve",
    "PTextAnswer",
    "PTextExtract",
    "BenchmarkRunner",
    "TEXT_PIPELINE_REGISTRY",
]
