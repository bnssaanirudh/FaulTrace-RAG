import hashlib
from datetime import date

from pydantic import BaseModel, Field


class EdgarFact(BaseModel):
    """Canonical model for a structured EDGAR/XBRL filing fact."""

    record_id: str = Field(default="", description="Unique record identifier")
    world_id: str = Field(default="", description="World this fact belongs to")
    cik: str = Field(..., description="Company identifier (CIK)")
    accession_number: str = Field(..., description="Filing accession number")
    filing_date: date = Field(..., description="Filing date")
    form_type: str = Field(..., description="Form type, e.g., 10-K, 10-Q")
    fiscal_year: int = Field(..., description="Fiscal year of the report")
    fiscal_period: str = Field(..., description="Fiscal period (e.g., FY, Q1, Q2)")

    # XBRL specific
    tag: str = Field(..., description="XBRL tag name, e.g., Revenues, Assets")
    namespace: str = Field(..., description="Taxonomy namespace, e.g., us-gaap, ifrs-full")
    unit: str = Field(..., description="Unit of measurement, e.g., USD, shares")
    value: float = Field(..., description="Numeric value of the fact")
    decimals: int | None = Field(None, description="Decimals/scale metadata")

    # Context dates
    start_date: date | None = Field(None, description="Start date for duration facts")
    end_date: date = Field(..., description="End date or instant date")

    segment_id: str | None = Field(None, description="Segment or context identifier, if any")
    source_url: str | None = Field(None, description="Source URL or local reference")

    raw_payload_hash: str = Field(..., description="Hash of the raw XBRL JSON/XML payload")
    canonical_fact_hash: str = Field(..., description="Hash of the canonicalized fact tuple")

    def generate_canonical_hash(self) -> str:
        """Deterministically hashes the core identity of this fact."""
        tuple_str = f"{self.cik}:{self.accession_number}:{self.tag}:{self.namespace}:{self.unit}:{self.end_date}:{self.segment_id}"
        return hashlib.sha256(tuple_str.encode("utf-8")).hexdigest()

    def set_ids(self, seed: int, index: int) -> None:
        """Sets deterministic IDs for world building."""
        self.world_id = f"edgar_s{seed}"
        self.record_id = f"edf_{seed:04d}_{index:08d}"


class EdgarCompanyFacts(BaseModel):
    """A collection of facts for a given company."""

    cik: str
    entity_name: str
    facts: list[EdgarFact]
