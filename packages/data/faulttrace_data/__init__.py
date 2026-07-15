"""
faulttrace_data: Deterministic Track M corpus generator.

Creates realistic Amazon review metadata distributions with controlled
edge cases for benchmarking LLM analytics pipelines.
"""

from faulttrace_data.generator import TrackMGenerator, WorldManifest
from faulttrace_data.generator import GENERATOR_VERSION

__all__ = ["TrackMGenerator", "WorldManifest", "GENERATOR_VERSION"]
__version__ = "0.1.0"
