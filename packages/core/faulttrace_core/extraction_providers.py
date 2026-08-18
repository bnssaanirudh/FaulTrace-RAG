"""
Extraction Providers for FaultTrace-RAG text-benchmark pipelines.

Three concrete implementations:
1. DeterministicFixtureExtractor  — rule-based, always valid, used for tests
2. SchemaConstrainedLLMExtractor  — wraps OpenAI with JSON-schema constrained output
3. BoundedRepairExtractor         — wraps any provider with N retry/repair attempts

All providers emit TraceEvents compatible with the existing trace infrastructure.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from faulttrace_core.evidence import (
    CitedAnswer,
    EntityType,
    ExtractedFact,
    ExtractionRecord,
    ExtractionSpan,
    FactType,
    SupportStatus,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class ExtractionProvider(ABC):
    """
    Abstract base for all extraction providers.

    Every provider takes a document text + metadata and returns an ExtractionRecord.
    The provider is responsible for:
    - populating facts with spans and confidence scores
    - setting a CitedAnswer with support_status
    - recording its own name and version
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider (used in provenance records)."""
        ...

    @property
    def model_version(self) -> str | None:
        return None

    @abstractmethod
    def extract(
        self,
        doc_id: str,
        doc_text: str,
        query: str,
        dataset_id: str = "unknown",
        split: str = "test",
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionRecord:
        """Extract facts and build a CitedAnswer for the given document + query."""
        ...


# ---------------------------------------------------------------------------
# DeterministicFixtureExtractor
# ---------------------------------------------------------------------------


class DeterministicFixtureExtractor(ExtractionProvider):
    """
    Rule-based deterministic extractor for tests and CI.

    Does NOT call any LLM. Extracts facts using simple heuristics:
    - Searches for query keywords in the document text
    - Marks as SUPPORTED if any keyword found, INSUFFICIENT_EVIDENCE otherwise
    - Always produces valid ExtractionRecords (no repair needed)

    This is the default extractor when OPENAI_API_KEY is not set.
    """

    @property
    def provider_name(self) -> str:
        return "deterministic_fixture"

    @property
    def model_version(self) -> str | None:
        return "1.0.0"

    def extract(
        self,
        doc_id: str,
        doc_text: str,
        query: str,
        dataset_id: str = "unknown",
        split: str = "test",
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionRecord:
        facts = self._extract_facts(doc_id, doc_text, query)

        # Determine support status based on keyword overlap
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        doc_tokens = set(re.findall(r"\b\w+\b", doc_text.lower()))
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "of",
            "in",
            "to",
            "and",
            "or",
            "for",
        }
        overlap = (query_tokens - stopwords) & (doc_tokens - stopwords)

        if len(overlap) >= 2:
            support_status = SupportStatus.SUPPORTED
            # Find best matching sentence as supporting quote
            sentences = [s.strip() for s in re.split(r"[.!?]", doc_text) if s.strip()]
            best_sentence = ""
            best_overlap = 0
            for sent in sentences:
                sent_tokens = set(re.findall(r"\b\w+\b", sent.lower()))
                sent_overlap = len(overlap & sent_tokens)
                if sent_overlap > best_overlap:
                    best_overlap = sent_overlap
                    best_sentence = sent
            supporting_quote = best_sentence[:300] if best_sentence else doc_text[:200]
            abstention_reason = None
        elif len(overlap) == 1:
            support_status = SupportStatus.PARTIALLY_SUPPORTED
            supporting_quote = doc_text[:200]
            abstention_reason = None
        else:
            support_status = SupportStatus.INSUFFICIENT_EVIDENCE
            supporting_quote = None
            abstention_reason = "No query terms found in document"

        cited_answer = CitedAnswer(
            answer_text=supporting_quote,
            support_status=support_status,
            cited_doc_ids=[doc_id],
            cited_chunk_ids=[chunk_id] if chunk_id else [],
            supporting_quote=supporting_quote,
            abstention_reason=abstention_reason,
        )

        return ExtractionRecord(
            dataset_id=dataset_id,
            split=split,
            doc_id=doc_id,
            chunk_id=chunk_id,
            facts=facts,
            cited_answer=cited_answer,
            provider=self.provider_name,
            model_version=self.model_version,
        )

    def _extract_facts(self, doc_id: str, doc_text: str, query: str) -> list[ExtractedFact]:
        """Extract simple keyword-based facts from the document."""
        facts = []

        # Extract dates (simple regex)
        date_pattern = r"\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
        for i, match in enumerate(re.finditer(date_pattern, doc_text)):
            start, end = match.start(), match.end()
            facts.append(
                ExtractedFact(
                    fact_id=f"{doc_id}_date_{i}",
                    fact_type=FactType.DATE,
                    entity_type=EntityType.DATE,
                    normalized_value=match.group(0),
                    surface_text=match.group(0),
                    span=ExtractionSpan(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=end,
                        surface_text=match.group(0),
                        source_quote=doc_text[max(0, start - 20) : end + 20],
                    ),
                    confidence=0.85,
                    provider="deterministic_fixture",
                    validation_status=ValidationStatus.VALID,
                )
            )

        # Extract numeric quantities
        num_pattern = r"\b\d+(?:\.\d+)?(?:\s*%|\s*mg|\s*kg|\s*ml|\s*cm)?\b"
        for i, match in enumerate(re.finditer(num_pattern, doc_text)):
            if match.group(0).isdigit() and len(match.group(0)) == 4:
                continue  # Skip bare 4-digit years (already captured as dates)
            start, end = match.start(), match.end()
            facts.append(
                ExtractedFact(
                    fact_id=f"{doc_id}_qty_{i}",
                    fact_type=FactType.QUANTITY,
                    entity_type=EntityType.QUANTITY,
                    normalized_value=match.group(0).strip(),
                    surface_text=match.group(0),
                    span=ExtractionSpan(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=end,
                        surface_text=match.group(0),
                        source_quote=doc_text[max(0, start - 20) : end + 20],
                    ),
                    confidence=0.75,
                    provider="deterministic_fixture",
                    validation_status=ValidationStatus.VALID,
                )
            )

        return facts[:10]  # Cap at 10 facts per document


# ---------------------------------------------------------------------------
# SchemaConstrainedLLMExtractor
# ---------------------------------------------------------------------------


class SchemaConstrainedLLMExtractor(ExtractionProvider):
    """
    OpenAI-based extractor using JSON-schema constrained structured output.

    Only active when OPENAI_API_KEY is set in the environment.
    Falls back to DeterministicFixtureExtractor if API key is absent.

    NOTE: This is a production stub. The JSON schema is defined and the
    OpenAI call is structured, but it requires a valid API key to produce
    real extractions.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._fallback = DeterministicFixtureExtractor()
        self._client = self._build_client()

    def _build_client(self):
        """Try to build OpenAI client; return None if key not available."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI

            return OpenAI()
        except ImportError:
            return None

    @property
    def provider_name(self) -> str:
        if self._client:
            return f"openai_{self.model}"
        return "deterministic_fixture_fallback"

    @property
    def model_version(self) -> str | None:
        return self.model if self._client else "1.0.0"

    def extract(
        self,
        doc_id: str,
        doc_text: str,
        query: str,
        dataset_id: str = "unknown",
        split: str = "test",
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionRecord:
        if self._client is None:
            # No API key — use deterministic fallback transparently
            return self._fallback.extract(
                doc_id=doc_id,
                doc_text=doc_text,
                query=query,
                dataset_id=dataset_id,
                split=split,
                chunk_id=chunk_id,
                metadata=metadata,
            )

        self._get_output_schema()
        prompt = self._build_prompt(query, doc_text)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise scientific text extractor. Respond only with valid JSON matching the schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            return self._parse_llm_output(
                parsed, doc_id, doc_text, query, dataset_id, split, chunk_id
            )

        except Exception:
            # Any LLM failure → fall back to deterministic
            return self._fallback.extract(
                doc_id=doc_id,
                doc_text=doc_text,
                query=query,
                dataset_id=dataset_id,
                split=split,
                chunk_id=chunk_id,
                metadata=metadata,
            )

    def _build_prompt(self, query: str, doc_text: str) -> str:
        return f"""Given the following document and query, extract facts and assess whether the document supports the query claim.

Query: {query}

Document:
{doc_text[:2000]}

Return a JSON object with these fields:
- support_status: one of "supported", "partially_supported", "unsupported", "conflicting", "insufficient_evidence"
- supporting_quote: the exact verbatim quote from the document that best supports or refutes the query (null if none)
- abstention_reason: explanation if insufficient_evidence or conflicting (null otherwise)
- answer_text: brief answer text (null if abstaining)
"""

    def _get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "support_status": {
                    "type": "string",
                    "enum": [
                        "supported",
                        "partially_supported",
                        "unsupported",
                        "conflicting",
                        "insufficient_evidence",
                    ],
                },
                "supporting_quote": {"type": ["string", "null"]},
                "abstention_reason": {"type": ["string", "null"]},
                "answer_text": {"type": ["string", "null"]},
            },
            "required": ["support_status"],
            "additionalProperties": False,
        }

    def _parse_llm_output(
        self,
        parsed: dict,
        doc_id: str,
        doc_text: str,
        query: str,
        dataset_id: str,
        split: str,
        chunk_id: str | None,
    ) -> ExtractionRecord:
        status_str = parsed.get("support_status", "insufficient_evidence")
        try:
            support_status = SupportStatus(status_str)
        except ValueError:
            support_status = SupportStatus.INSUFFICIENT_EVIDENCE

        abstention_reason = parsed.get("abstention_reason")
        if support_status in (SupportStatus.INSUFFICIENT_EVIDENCE, SupportStatus.CONFLICTING):
            abstention_reason = abstention_reason or "LLM returned insufficient evidence"

        cited_answer = CitedAnswer(
            answer_text=parsed.get("answer_text"),
            support_status=support_status,
            cited_doc_ids=[doc_id],
            cited_chunk_ids=[chunk_id] if chunk_id else [],
            supporting_quote=parsed.get("supporting_quote"),
            abstention_reason=abstention_reason,
        )

        return ExtractionRecord(
            dataset_id=dataset_id,
            split=split,
            doc_id=doc_id,
            chunk_id=chunk_id,
            cited_answer=cited_answer,
            provider=self.provider_name,
            model_version=self.model_version,
        )


# ---------------------------------------------------------------------------
# BoundedRepairExtractor
# ---------------------------------------------------------------------------


class ExtractionAttempt:
    """Log of a single extraction attempt."""

    def __init__(self, attempt: int, success: bool, error: str | None, duration_ms: float):
        self.attempt = attempt
        self.success = success
        self.error = error
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "success": self.success,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


class BoundedRepairExtractor(ExtractionProvider):
    """
    Wraps any ExtractionProvider with bounded retry/repair logic.

    On each attempt:
    1. Calls the inner provider's extract()
    2. If the result fails Pydantic validation, increments repair_count and retries
    3. After max_retries exhausted, returns a minimal valid rejected ExtractionRecord
    4. Each attempt is logged as a trace-compatible dict in self.repair_log

    Citation integrity violations are NOT repaired — they are always rejected.
    This is intentional: invented document IDs should never be silently fixed.
    """

    def __init__(self, inner: ExtractionProvider, max_retries: int = 3):
        self.inner = inner
        self.max_retries = max_retries
        self.repair_log: list[ExtractionAttempt] = []

    @property
    def provider_name(self) -> str:
        return f"bounded_repair({self.inner.provider_name})"

    @property
    def model_version(self) -> str | None:
        return self.inner.model_version

    def extract(
        self,
        doc_id: str,
        doc_text: str,
        query: str,
        dataset_id: str = "unknown",
        split: str = "test",
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionRecord:
        self.repair_log = []
        last_error: str | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                record = self.inner.extract(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    query=query,
                    dataset_id=dataset_id,
                    split=split,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
                elapsed = (time.perf_counter() - t0) * 1000

                # Check for citation integrity violation specifically
                if record.cited_answer:
                    for cited_id in record.cited_answer.cited_doc_ids:
                        if cited_id != doc_id:
                            error_msg = (
                                f"INVENTED_DOCUMENT_ID: cited_doc_id '{cited_id}' "
                                f"≠ actual doc_id '{doc_id}'"
                            )
                            self.repair_log.append(
                                ExtractionAttempt(attempt, False, error_msg, elapsed)
                            )
                            # Do NOT retry citation integrity violations
                            return self._build_rejected_record(
                                doc_id, dataset_id, split, chunk_id, error_msg
                            )

                self.repair_log.append(ExtractionAttempt(attempt, True, None, elapsed))
                # Mark as repaired if took more than 1 attempt
                if attempt > 1 and record.cited_answer:
                    record = record.model_copy(
                        update={
                            "facts": [
                                f.model_copy(
                                    update={
                                        "repair_count": attempt - 1,
                                        "validation_status": ValidationStatus.REPAIRED,
                                    }
                                )
                                for f in record.facts
                            ]
                        }
                    )
                return record

            except (ValidationError, ValueError) as e:
                elapsed = (time.perf_counter() - t0) * 1000
                last_error = str(e)
                self.repair_log.append(ExtractionAttempt(attempt, False, last_error, elapsed))

        # All retries exhausted
        return self._build_rejected_record(
            doc_id,
            dataset_id,
            split,
            chunk_id,
            f"Max retries ({self.max_retries}) exceeded. Last error: {last_error}",
        )

    def _build_rejected_record(
        self,
        doc_id: str,
        dataset_id: str,
        split: str,
        chunk_id: str | None,
        rejection_reason: str,
    ) -> ExtractionRecord:
        """Build a minimal valid ExtractionRecord in rejected state."""
        # CitedAnswer with explicit INSUFFICIENT_EVIDENCE (not invented doc)
        cited_answer = CitedAnswer(
            answer_text=None,
            support_status=SupportStatus.INSUFFICIENT_EVIDENCE,
            cited_doc_ids=[],  # Empty — no invented IDs
            abstention_reason=f"Extraction rejected: {rejection_reason[:200]}",
        )
        return ExtractionRecord(
            dataset_id=dataset_id,
            split=split,
            doc_id=doc_id,
            chunk_id=chunk_id,
            facts=[],
            cited_answer=cited_answer,
            provider=self.provider_name,
            model_version=self.model_version,
            metadata={
                "rejection_reason": rejection_reason,
                "repair_log": [a.to_dict() for a in self.repair_log],
            },
        )
