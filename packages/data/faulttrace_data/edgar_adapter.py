import json
from datetime import date
from pathlib import Path

from faulttrace_core.edgar_models import EdgarCompanyFacts, EdgarFact


class EdgarAdapter:
    """Ingests local SEC EDGAR XBRL JSON payloads into canonical structured facts.
    Respects SEC guidelines by only doing local operations (no live scraping without explicit override).
    """

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir

    def list_available_ciks(self) -> list[str]:
        """Lists CIKs available in the local fixtures directory."""
        ciks = []
        if self.fixtures_dir.exists():
            for f in self.fixtures_dir.glob("*.json"):
                ciks.append(f.stem.replace("mock_", "").upper())
        return ciks

    def load_company_facts(self, cik: str) -> EdgarCompanyFacts | None:
        """Loads and normalizes facts for a single company."""
        file_path = self.fixtures_dir / f"mock_{cik.lower()}.json"
        if not file_path.exists():
            return None

        with open(file_path) as f:
            raw_data = json.load(f)

        entity_name = raw_data.get("entityName", f"Company {cik}")
        raw_facts = raw_data.get("facts", {})

        normalized_facts: list[EdgarFact] = []

        # Parse us-gaap and ifrs-full facts
        for namespace, tags in raw_facts.items():
            for tag, fact_data in tags.items():
                units = fact_data.get("units", {})
                for unit, measurements in units.items():
                    for m in measurements:
                        # Extract core properties
                        end = m.get("end")
                        start = m.get("start")
                        val = m.get("val")
                        acc = m.get("accn", "0000000000-00-000000")
                        form = m.get("form", "10-K")
                        fy = m.get("fy", 0)
                        fp = m.get("fp", "FY")

                        if not end or val is None:
                            continue

                        # Convert string dates to datetime.date
                        try:
                            end_date = date.fromisoformat(end)
                            start_date = date.fromisoformat(start) if start else None
                        except ValueError:
                            continue

                        import hashlib
                        raw_hash = hashlib.sha256(json.dumps(m, sort_keys=True).encode("utf-8")).hexdigest()

                        fact = EdgarFact(
                            cik=cik,
                            accession_number=acc,
                            filing_date=end_date, # Approximation for fixture simplicity
                            form_type=form,
                            fiscal_year=fy,
                            fiscal_period=fp,
                            tag=tag,
                            namespace=namespace,
                            unit=unit,
                            value=float(val),
                            decimals=None,
                            start_date=start_date,
                            end_date=end_date,
                            raw_payload_hash=raw_hash,
                            canonical_fact_hash="" # Placeholder to be computed
                        )
                        fact.canonical_fact_hash = fact.generate_canonical_hash()
                        normalized_facts.append(fact)

        return EdgarCompanyFacts(
            cik=cik,
            entity_name=entity_name,
            facts=normalized_facts
        )
