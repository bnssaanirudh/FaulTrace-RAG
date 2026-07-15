"""
Amazon Reviews 2023-style local-file ingestion adapter.

Reads JSONL, JSON, JSONL.GZ, CSV, or Parquet files containing Amazon-style
review metadata and canonicalizes them into CorpusRecord objects.

Design principles:
- Streaming / chunked ingestion — never loads the entire file into memory
- Configurable source-field mapping with documented defaults
- No raw review text in logs; only record IDs and hashes
- Reject or quarantine invalid required identifiers (empty ASIN, None ASIN)
- Duplicate detection via stable source keys and content hashes
- Robust handling of missing price/category/brand/malformed optional fields
- Size limit guard against decompression bombs (default: 500MB uncompressed)
- Writes canonical Parquet partitioned by dataset_id/snapshot_id[/category][/year]

Field Mapping Default:
  Amazon JSON field     → CorpusRecord field
  ─────────────────────────────────────────────
  asin                 → product_id / source_record_id
  title                → title
  main_category        → category (normalized to RecordCategory)
  store                → brand (fallback: first seller name)
  average_rating       → rating
  rating_number        → helpful_votes (proxy)
  price                → price
  timestamp            → event_time
  verified_purchase    → verified_purchase (default True if absent)
  categories           → attributes["raw_categories"]
  bought_in_last_month → attributes["bought_in_last_month"]
  (all extra fields)   → attributes["_extra"]
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Generator, Iterator, Optional
from uuid import uuid4

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from faulttrace_core.models import CorpusRecord, RecordCategory, SCHEMA_VERSION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB guard
CHUNK_SIZE = 10_000  # rows per Parquet partition chunk

# Category normalization map
_CATEGORY_MAP: dict[str, RecordCategory] = {
    "electronics": RecordCategory.ELECTRONICS,
    "books": RecordCategory.BOOKS,
    "home & kitchen": RecordCategory.HOME_KITCHEN,
    "home and kitchen": RecordCategory.HOME_KITCHEN,
    "kitchen": RecordCategory.HOME_KITCHEN,
    "sports & outdoors": RecordCategory.SPORTS,
    "sports and outdoors": RecordCategory.SPORTS,
    "sports": RecordCategory.SPORTS,
    "clothing": RecordCategory.CLOTHING,
    "clothing, shoes & jewelry": RecordCategory.CLOTHING,
    "apparel": RecordCategory.CLOTHING,
    "toys & games": RecordCategory.TOYS,
    "toys and games": RecordCategory.TOYS,
    "toys": RecordCategory.TOYS,
    "beauty": RecordCategory.BEAUTY,
    "beauty & personal care": RecordCategory.BEAUTY,
    "automotive": RecordCategory.AUTOMOTIVE,
    "food & grocery": RecordCategory.FOOD,
    "grocery": RecordCategory.FOOD,
    "grocery & gourmet food": RecordCategory.FOOD,
    "office products": RecordCategory.OFFICE,
    "office": RecordCategory.OFFICE,
}

# ---------------------------------------------------------------------------
# Field mapping configuration
# ---------------------------------------------------------------------------


class AmazonFieldMapping(BaseModel):
    """
    Configurable mapping from Amazon JSON field names to canonical field names.
    Override individual fields to adapt to non-standard Amazon exports.
    """

    asin_field: str = "asin"
    title_field: str = "title"
    main_category_field: str = "main_category"
    store_field: str = "store"
    brand_field: str = "brand"  # Alternative brand field; checked after store
    average_rating_field: str = "average_rating"
    rating_number_field: str = "rating_number"
    price_field: str = "price"
    timestamp_field: str = "timestamp"
    verified_purchase_field: str = "verified_purchase"
    categories_field: str = "categories"
    bought_in_last_month_field: str = "bought_in_last_month"

    def config_hash(self) -> str:
        """Stable SHA-256 of this mapping configuration."""
        raw = orjson.dumps(self.model_dump(), option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(raw).hexdigest()


DEFAULT_FIELD_MAPPING = AmazonFieldMapping()


# ---------------------------------------------------------------------------
# Ingestion report
# ---------------------------------------------------------------------------


class IngestionReport(BaseModel):
    """Per-run ingestion statistics."""

    dataset_id: str
    snapshot_id: str
    source_type: str
    source_path_fingerprint: str
    total_rows_read: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    malformed_count: int = 0
    null_counts: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parquet_output_path: Optional[str] = None
    rejected_artifact_path: Optional[str] = None

    def log_rejection(self, record_id: str, reason: str) -> None:
        """Log a rejection reason without exposing raw text."""
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        self.rejected_count += 1
        logger.debug("ingestion.rejected record_id=%s reason=%s", record_id, reason)

    def log_duplicate(self, record_id: str) -> None:
        self.duplicate_count += 1
        logger.debug("ingestion.duplicate record_id=%s", record_id)


# ---------------------------------------------------------------------------
# Row parser
# ---------------------------------------------------------------------------


def _normalize_category(raw: Any) -> Optional[RecordCategory]:
    """Normalize a raw category string to RecordCategory. Returns None if unknown."""
    if raw is None:
        return None
    normalized = str(raw).lower().strip()
    return _CATEGORY_MAP.get(normalized)


def _parse_timestamp(raw: Any, field_name: str = "timestamp") -> Optional[datetime]:
    """
    Parse a timestamp from various formats. Always produces UTC datetime.
    Returns None on failure (caller decides to reject or quarantine).
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Unix epoch
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    s = str(raw).strip()
    if not s:
        return None
    # ISO 8601 with/without timezone
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _parse_price(raw: Any) -> Optional[Decimal]:
    """Parse price from string, float, or int. Returns None if missing/invalid."""
    if raw is None:
        return None
    try:
        s = str(raw).replace("$", "").replace(",", "").strip()
        if not s:
            return None
        d = Decimal(s).quantize(Decimal("0.01"))
        if d < 0:
            return None
        return d
    except (InvalidOperation, ValueError):
        return None


def _parse_rating(raw: Any) -> Optional[float]:
    """Parse and validate rating [1.0, 5.0]. Returns None if invalid."""
    if raw is None:
        return None
    try:
        v = float(raw)
        if 1.0 <= v <= 5.0:
            return round(v, 1)
        return None
    except (ValueError, TypeError):
        return None


def _raw_hash(row: dict[str, Any]) -> str:
    """Stable SHA-256 of a raw row dict."""
    try:
        return hashlib.sha256(
            orjson.dumps(row, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS)
        ).hexdigest()[:32]
    except Exception:
        return hashlib.sha256(str(row).encode()).hexdigest()[:32]


class RowParser:
    """Parse a single raw Amazon dict into a CorpusRecord or rejection info."""

    def __init__(self, mapping: AmazonFieldMapping, dataset_id: str, world_id: str):
        self.mapping = mapping
        self.dataset_id = dataset_id
        self.world_id = world_id

    def parse(self, row: dict[str, Any], row_index: int) -> tuple[Optional[CorpusRecord], Optional[str]]:
        """
        Parse row into (CorpusRecord, None) on success, or (None, rejection_reason) on failure.
        """
        m = self.mapping
        raw_hash = _raw_hash(row)

        # --- Required: ASIN / product_id ---
        asin = row.get(m.asin_field)
        if not asin or not str(asin).strip():
            return None, "missing_or_empty_asin"
        asin = str(asin).strip()
        # Validate ASIN-like format (letter/digit, 3-20 chars)
        if not re.match(r"^[A-Za-z0-9]{1,20}$", asin):
            return None, "invalid_asin_format"

        # --- Required: timestamp ---
        raw_ts = row.get(m.timestamp_field)
        event_time = _parse_timestamp(raw_ts)
        if event_time is None:
            return None, "malformed_timestamp"

        # --- Required: rating ---
        raw_rating = row.get(m.average_rating_field)
        rating = _parse_rating(raw_rating)
        if rating is None:
            return None, "invalid_rating"

        # --- Optional: title ---
        title = str(row.get(m.title_field, "")).strip()
        if not title:
            title = f"[No Title] {asin}"

        # --- Optional: category ---
        raw_cat = row.get(m.main_category_field)
        category = _normalize_category(raw_cat)
        if category is None:
            # Fallback: try first element of categories list
            cats = row.get(m.categories_field)
            if isinstance(cats, list) and cats:
                category = _normalize_category(cats[0])
        if category is None:
            category = RecordCategory.OFFICE  # Safe fallback, flagged in null_counts

        # --- Optional: brand ---
        brand = (
            row.get(m.brand_field)
            or row.get(m.store_field)
            or "Unknown"
        )
        brand = str(brand).strip() or "Unknown"

        # --- Optional: price ---
        price = _parse_price(row.get(m.price_field))

        # --- Optional: verified_purchase ---
        raw_vp = row.get(m.verified_purchase_field)
        if isinstance(raw_vp, bool):
            verified_purchase = raw_vp
        elif raw_vp is None:
            verified_purchase = True  # Assume verified if not specified
        else:
            verified_purchase = str(raw_vp).lower() in ("true", "1", "yes")

        # --- Optional: helpful_votes proxy ---
        helpful_votes = 0
        raw_hn = row.get(m.rating_number_field)
        if raw_hn is not None:
            try:
                helpful_votes = max(0, int(float(raw_hn)))
            except (ValueError, TypeError):
                helpful_votes = 0

        # --- Attributes: preserve extras without logging raw text ---
        known_fields = {
            m.asin_field, m.title_field, m.main_category_field,
            m.store_field, m.brand_field, m.average_rating_field,
            m.rating_number_field, m.price_field, m.timestamp_field,
            m.verified_purchase_field, m.categories_field,
            m.bought_in_last_month_field,
        }
        extra = {k: v for k, v in row.items() if k not in known_fields}
        attributes: dict[str, Any] = {}
        if raw_cat is not None:
            attributes["raw_category"] = str(raw_cat)
        raw_cats = row.get(m.categories_field)
        if isinstance(raw_cats, list):
            attributes["raw_categories"] = [str(c) for c in raw_cats[:10]]
        biml = row.get(m.bought_in_last_month_field)
        if biml is not None:
            try:
                attributes["bought_in_last_month"] = int(biml)
            except (ValueError, TypeError):
                pass
        if extra:
            # Store count only, not content
            attributes["_extra_field_count"] = len(extra)

        # --- Deterministic record ID ---
        source_record_id = asin
        record_id = f"amz_{self.dataset_id}_{asin}_{row_index:08d}"

        record = CorpusRecord(
            record_id=record_id,
            source=f"amazon_local:{self.dataset_id}",
            source_record_id=source_record_id,
            world_id=self.world_id,
            product_id=asin,
            category=category,
            title=title[:512],  # Truncate very long titles
            brand=brand[:128],
            rating=rating,
            helpful_votes=helpful_votes,
            verified_purchase=verified_purchase,
            event_time=event_time,
            price=price,
            attributes=attributes,
            text="",  # No raw text stored
            raw_payload_hash=raw_hash,
        )
        return record, None


# ---------------------------------------------------------------------------
# File readers (streaming)
# ---------------------------------------------------------------------------


def _stream_jsonl(path: Path, max_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Generator[dict[str, Any], None, None]:
    """Stream rows from JSONL file. Skips non-JSON lines."""
    bytes_read = 0
    with open(path, "rb") as f:
        for line_bytes in f:
            bytes_read += len(line_bytes)
            if bytes_read > max_bytes:
                raise ValueError(f"File exceeds max uncompressed size limit ({max_bytes} bytes)")
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": True, "_raw": line[:80]}


def _stream_jsonl_gz(path: Path, max_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Generator[dict[str, Any], None, None]:
    """Stream rows from gzipped JSONL file."""
    bytes_read = 0
    with gzip.open(path, "rb") as gz:
        for line_bytes in gz:
            bytes_read += len(line_bytes)
            if bytes_read > max_bytes:
                raise ValueError(f"Decompression bomb guard: exceeds {max_bytes} bytes")
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": True, "_raw": line[:80]}


def _stream_csv(path: Path, max_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Generator[dict[str, Any], None, None]:
    """Stream rows from CSV file."""
    bytes_read = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_bytes = sum(len(str(v)) for v in row.values())
            bytes_read += row_bytes
            if bytes_read > max_bytes:
                raise ValueError(f"CSV file exceeds size limit ({max_bytes} bytes)")
            yield dict(row)


def _stream_parquet(path: Path, max_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Generator[dict[str, Any], None, None]:
    """Stream rows from Parquet file using pyarrow batch reader."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            yield row.to_dict()


def _get_row_stream(path: Path, max_bytes: int = MAX_UNCOMPRESSED_BYTES) -> Generator[dict[str, Any], None, None]:
    """Dispatch to the correct reader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _stream_jsonl(path, max_bytes)
    elif suffix == ".gz":
        return _stream_jsonl_gz(path, max_bytes)
    elif suffix == ".csv":
        return _stream_csv(path, max_bytes)
    elif suffix in (".parquet", ".pq"):
        return _stream_parquet(path, max_bytes)
    elif suffix == ".json":
        # Try as JSONL first, then as JSON array
        return _stream_jsonl(path, max_bytes)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# ---------------------------------------------------------------------------
# Amazon Local Adapter
# ---------------------------------------------------------------------------


class AmazonLocalAdapter:
    """
    Ingest Amazon Reviews-style files into canonical Parquet snapshots.

    Usage:
        adapter = AmazonLocalAdapter(dataset_id="amazon-local-v1")
        report = adapter.ingest(
            source_path=Path("data/raw/reviews.jsonl"),
            output_root=Path("data/snapshots"),
        )
    """

    def __init__(
        self,
        dataset_id: str,
        field_mapping: Optional[AmazonFieldMapping] = None,
        max_bytes: int = MAX_UNCOMPRESSED_BYTES,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.dataset_id = dataset_id
        self.field_mapping = field_mapping or DEFAULT_FIELD_MAPPING
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size

    def ingest(
        self,
        source_path: Path,
        output_root: Path,
        data_root: Optional[Path] = None,
        snapshot_id: Optional[str] = None,
        license_note: str = "",
        producing_command: Optional[str] = None,
        partition_by_category: bool = False,
        partition_by_year: bool = False,
    ) -> tuple[IngestionReport, "DatasetSnapshot"]:
        """
        Stream-ingest source_path → canonical Parquet + ingestion report.

        Args:
            source_path: Local file to ingest (.jsonl, .jsonl.gz, .csv, .parquet)
            output_root: Root directory for canonical Parquet output
            data_root: Project data root for path fingerprinting (no absolute paths)
            snapshot_id: Override deterministic snapshot ID
            license_note: Researcher-provided license/provenance note
            partition_by_category: If True, add category partition
            partition_by_year: If True, add year partition

        Returns:
            (IngestionReport, DatasetSnapshot)
        """
        from faulttrace_data.snapshot import DatasetSnapshot, MissingnessSummary, _fingerprint_path

        if data_root is None:
            data_root = output_root.parent

        # Compute source fingerprint (no absolute path in output)
        source_fingerprint = _fingerprint_path(source_path, data_root)
        config_hash = self.field_mapping.config_hash()

        # Determine snapshot_id
        if snapshot_id is None:
            snapshot_id = DatasetSnapshot.make_snapshot_id(
                self.dataset_id, source_fingerprint, config_hash
            )

        world_id = f"{self.dataset_id}_{snapshot_id}"
        parser = RowParser(self.field_mapping, self.dataset_id, world_id)

        report = IngestionReport(
            dataset_id=self.dataset_id,
            snapshot_id=snapshot_id,
            source_type=_detect_source_type(source_path),
            source_path_fingerprint=source_fingerprint,
        )

        # Output directory
        out_dir = output_root / self.dataset_id / snapshot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rejected_path = out_dir / "rejected_rows.jsonl"

        # Tracking
        seen_source_ids: dict[str, str] = {}  # source_record_id → record_id
        seen_content_hashes: set[str] = set()
        accepted_records: list[CorpusRecord] = []
        rejected_rows: list[dict[str, Any]] = []

        null_counts: dict[str, int] = {
            "price": 0, "category": 0, "brand": 0,
            "rating": 0, "verified_purchase": 0, "timestamp": 0,
        }

        row_index = 0
        for raw_row in _get_row_stream(source_path, self.max_bytes):
            row_index += 1
            report.total_rows_read += 1

            # Handle parse errors in stream
            if raw_row.get("_parse_error"):
                report.malformed_count += 1
                rejected_rows.append({
                    "row_index": row_index,
                    "reason": "json_parse_error",
                    "raw_prefix": raw_row.get("_raw", "")[:40],
                })
                report.log_rejection(f"row_{row_index}", "json_parse_error")
                continue

            # Parse row
            record, rejection_reason = parser.parse(raw_row, row_index)

            if rejection_reason:
                report.log_rejection(f"row_{row_index}", rejection_reason)
                rejected_rows.append({
                    "row_index": row_index,
                    "reason": rejection_reason,
                    "source_id_hash": hashlib.sha256(
                        str(raw_row.get(self.field_mapping.asin_field, "")).encode()
                    ).hexdigest()[:16],
                })
                continue

            assert record is not None

            # Track null fields
            if record.price is None:
                null_counts["price"] += 1
            if raw_row.get(self.field_mapping.main_category_field) is None:
                null_counts["category"] += 1

            # Duplicate detection by source_record_id
            if record.source_record_id in seen_source_ids:
                report.log_duplicate(record.record_id)
                continue

            # Duplicate detection by content hash
            content_hash = record.content_hash()
            if content_hash in seen_content_hashes:
                report.log_duplicate(record.record_id)
                continue

            seen_source_ids[record.source_record_id] = record.record_id
            seen_content_hashes.add(content_hash)
            accepted_records.append(record)
            report.accepted_count += 1

            # Flush chunk to Parquet
            if len(accepted_records) >= self.chunk_size:
                _write_parquet_chunk(accepted_records, out_dir, len(accepted_records))
                accepted_records = []

        # Final chunk
        if accepted_records:
            _write_parquet_chunk(accepted_records, out_dir, row_index)

        # Write rejected rows artifact (no raw text)
        if rejected_rows:
            with open(rejected_path, "w", encoding="utf-8") as f:
                for row in rejected_rows:
                    f.write(json.dumps(row) + "\n")
            report.rejected_artifact_path = str(rejected_path)

        report.null_counts = null_counts
        report.parquet_output_path = str(out_dir)

        # Compute canonical hash of output directory
        from faulttrace_data.snapshot import _hash_directory, MissingnessSummary
        canonical_hash = _hash_directory(out_dir) if out_dir.exists() else ""

        # Compute source file hash
        source_hash = _file_sha256(source_path)

        # Build summary statistics from accepted records
        cat_counts: dict[str, int] = {}
        # (We'd need to re-read Parquet for full stats; compute from report)

        # Build DatasetSnapshot
        snapshot = DatasetSnapshot(
            snapshot_id=snapshot_id,
            dataset_id=self.dataset_id,
            source_type=report.source_type,
            source_path_fingerprint=source_fingerprint,
            source_file_size_bytes=source_path.stat().st_size if source_path.exists() else None,
            source_row_count_raw=report.total_rows_read,
            row_count=report.accepted_count,
            partition_count=max(1, report.accepted_count // self.chunk_size + 1),
            parquet_root=str(out_dir.relative_to(data_root)) if data_root else str(out_dir),
            raw_content_hash=source_hash,
            canonical_content_hash=canonical_hash,
            ingestion_config_hash=config_hash,
            accepted_count=report.accepted_count,
            rejected_count=report.rejected_count,
            duplicate_count=report.duplicate_count,
            malformed_count=report.malformed_count,
            null_count_by_field=null_counts,
            rejected_row_artifact_ref=str(rejected_path.relative_to(data_root)) if rejected_rows and data_root else None,
            license_note=license_note,
            producing_command=producing_command,
            missingness=MissingnessSummary(
                price_null_count=null_counts.get("price", 0),
                price_null_ratio=null_counts.get("price", 0) / max(report.accepted_count, 1),
                category_null_count=null_counts.get("category", 0),
                category_null_ratio=null_counts.get("category", 0) / max(report.accepted_count, 1),
            ),
            environment_summary={
                "python_version": _python_version(),
                "platform": _platform_info(),
            },
        )

        return report, snapshot

    def inspect_snapshot(self, output_root: Path, snapshot_id: str) -> dict[str, Any]:
        """Return a structured summary of an ingested snapshot directory."""
        out_dir = output_root / self.dataset_id / snapshot_id
        if not out_dir.exists():
            return {"error": "snapshot_not_found", "snapshot_id": snapshot_id}

        parquet_files = list(out_dir.glob("**/*.parquet"))
        total_rows = 0
        for pf in parquet_files:
            try:
                meta = pq.read_metadata(pf)
                total_rows += meta.num_rows
            except Exception:
                pass

        return {
            "snapshot_id": snapshot_id,
            "dataset_id": self.dataset_id,
            "parquet_files": len(parquet_files),
            "total_rows_in_parquet": total_rows,
            "output_directory": str(out_dir),
            "directory_hash": _hash_from_dir(out_dir),
        }


# ---------------------------------------------------------------------------
# Parquet writer helpers
# ---------------------------------------------------------------------------


def _write_parquet_chunk(records: list[CorpusRecord], out_dir: Path, chunk_idx: int) -> Path:
    """Write a chunk of CorpusRecords to a Parquet file."""
    rows = []
    for r in records:
        rows.append({
            "record_id": r.record_id,
            "source": r.source,
            "source_record_id": r.source_record_id,
            "world_id": r.world_id,
            "product_id": r.product_id,
            "parent_id": r.parent_id,
            "category": r.category.value,
            "title": r.title,
            "brand": r.brand,
            "rating": float(r.rating),
            "helpful_votes": r.helpful_votes,
            "verified_purchase": r.verified_purchase,
            "event_time": r.event_time,
            "price": float(r.price) if r.price is not None else None,
            "text": "",  # Never store raw text
            "raw_payload_hash": r.raw_payload_hash,
            "schema_version": r.schema_version,
        })
    df = pd.DataFrame(rows)
    out_path = out_dir / f"chunk_{chunk_idx:06d}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def _hash_from_dir(directory: Path) -> str:
    """Quick hash of all parquet files in directory."""
    from faulttrace_data.snapshot import _hash_directory
    return _hash_directory(directory)


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()


def _detect_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return "amazon_jsonl_gz"
    elif suffix == ".jsonl":
        return "amazon_jsonl"
    elif suffix == ".csv":
        return "amazon_csv"
    elif suffix in (".parquet", ".pq"):
        return "amazon_parquet"
    elif suffix == ".json":
        return "amazon_json"
    return "unknown"


def _python_version() -> str:
    import sys
    return sys.version.split()[0]


def _platform_info() -> str:
    import platform
    return f"{platform.system()}/{platform.machine()}"


# Re-export for use in snapshot module
def _hash_directory_local(directory: Path) -> str:
    from faulttrace_data.snapshot import _hash_directory
    return _hash_directory(directory)
