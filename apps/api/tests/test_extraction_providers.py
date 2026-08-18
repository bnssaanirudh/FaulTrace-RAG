import pytest
from faulttrace_core.extraction_providers import (
    DeterministicFixtureExtractor,
    SchemaConstrainedLLMExtractor,
    get_default_extractor,
)


def test_get_default_extractor_deterministic(monkeypatch):
    monkeypatch.delenv("EXTRACTION_PROVIDER", raising=False)
    extractor = get_default_extractor()
    assert isinstance(extractor, DeterministicFixtureExtractor)

def test_get_default_extractor_openai_without_key_raises(monkeypatch):
    monkeypatch.setenv("EXTRACTION_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    extractor = get_default_extractor()
    assert isinstance(extractor, SchemaConstrainedLLMExtractor)

    with pytest.raises(RuntimeError) as exc_info:
        extractor.extract(
            doc_id="d1",
            doc_text="Test document",
            query="Test query"
        )
    assert "EXTRACTION_PROVIDER=openai but OPENAI_API_KEY is not set" in str(exc_info.value)

def test_llm_extractor_validates_supporting_quote(monkeypatch):
    # Mock the openai client to return a response with invalid quote
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")

    extractor = SchemaConstrainedLLMExtractor()

    class MockChoice:
        class MockMessage:
            content = '{"support_status": "supported", "supporting_quote": "This is not in the text"}'
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    class MockCompletions:
        def create(self, *args, **kwargs):
            return MockResponse()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    extractor._client = MockClient()

    with pytest.raises(ValueError) as exc_info:
        extractor.extract(
            doc_id="d1",
            doc_text="Original source text does not contain that quote",
            query="Test query"
        )
    assert "Extracted supporting_quote is not an exact substring" in str(exc_info.value)

def test_llm_extractor_valid_supporting_quote(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key")

    extractor = SchemaConstrainedLLMExtractor()

    class MockChoice:
        class MockMessage:
            content = '{"support_status": "supported", "supporting_quote": "source text"}'
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    class MockCompletions:
        def create(self, *args, **kwargs):
            return MockResponse()

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    extractor._client = MockClient()

    record = extractor.extract(
        doc_id="d1",
        doc_text="Original source text does not contain that quote but wait it does have source text",
        query="Test query",
        dataset_id="test_ds",
        split="test",
    )
    assert record.cited_answer.support_status.value == "supported"
    assert record.cited_answer.supporting_quote == "source text"
