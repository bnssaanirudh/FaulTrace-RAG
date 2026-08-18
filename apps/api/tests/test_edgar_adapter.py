from pathlib import Path

from faulttrace_core.edgar_models import EdgarFact
from faulttrace_data.edgar_adapter import EdgarAdapter


def test_edgar_adapter_init():
    adapter = EdgarAdapter(fixtures_dir=Path("/tmp/fixtures"))
    assert str(adapter.fixtures_dir) == str(Path("/tmp/fixtures"))


def test_edgar_fact_validation():
    fact = EdgarFact(
        cik="0001234567",
        accession_number="0001234567-23-000012",
        filing_date="2023-12-31",
        form_type="10-K",
        fiscal_year=2023,
        fiscal_period="FY",
        tag="Revenues",
        namespace="us-gaap",
        unit="USD",
        value=1000000.0,
        start_date="2023-01-01",
        end_date="2023-12-31",
        raw_payload_hash="mock_hash",
        canonical_fact_hash="mock_canon_hash",
    )
    assert fact.cik == "0001234567"
    assert fact.form_type == "10-K"
