"""
Deterministic Track M corpus generator.

Generates realistic Amazon review metadata without downloading the actual dataset.
Records are seeded deterministically; for a fixed seed, records in N=10 are the
first N of N=50, which are the first N of N=200, etc. (nestedness guarantee).

Generator Version: update this when the generation algorithm changes.
Records generated from the same seed and version must be byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from faulttrace_core.models import (
    CorpusRecord,
    CorpusWorld,
    RecordCategory,
    SCHEMA_VERSION,
)

GENERATOR_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Reference data (no external dependencies)
# ---------------------------------------------------------------------------

_CATEGORIES_WITH_WEIGHTS = [
    (RecordCategory.ELECTRONICS, 0.20),
    (RecordCategory.BOOKS, 0.18),
    (RecordCategory.HOME_KITCHEN, 0.15),
    (RecordCategory.SPORTS, 0.10),
    (RecordCategory.CLOTHING, 0.12),
    (RecordCategory.TOYS, 0.08),
    (RecordCategory.BEAUTY, 0.07),
    (RecordCategory.AUTOMOTIVE, 0.04),
    (RecordCategory.FOOD, 0.04),
    (RecordCategory.OFFICE, 0.02),
]

_CATEGORY_NAMES = [c[0] for c in _CATEGORIES_WITH_WEIGHTS]
_CATEGORY_WEIGHTS = [c[1] for c in _CATEGORIES_WITH_WEIGHTS]

# Long-tail brand distribution: few dominant brands, many small ones
_BRANDS_BY_CATEGORY: dict[RecordCategory, list[str]] = {
    RecordCategory.ELECTRONICS: [
        "TechPrime", "VoltEdge", "ClearView", "DataSync", "NovaTech",
        "WireFlex", "PixelMax", "SoundWave", "GridCore", "ByteLeaf",
        "ZetaElec", "OmniLink", "BlueStar", "CrystalBit", "NeoPulse",
    ],
    RecordCategory.BOOKS: [
        "PenguinPress", "OakTree", "SilverPage", "MindVault", "LitGuild",
        "WordBridge", "StoryCore", "ChapterOne", "NovelNest", "PageTurn",
    ],
    RecordCategory.HOME_KITCHEN: [
        "HomeChef", "NestWare", "GreenLeaf", "CookPro", "TableCraft",
        "KitchenEdge", "PureHome", "AquaFlow", "WarmNest", "ClearCook",
    ],
    RecordCategory.SPORTS: [
        "SportPeak", "ActiveEdge", "TrailBlazer", "FitCore", "VeloMax",
        "IronGrip", "PeakPulse", "SwiftGear", "TerraFit", "PowerStride",
    ],
    RecordCategory.CLOTHING: [
        "StyleCraft", "WearWell", "UrbanThread", "SoftLayer", "ClothCo",
        "FabricEdge", "TrendMark", "CozyWear", "LinenLux", "CottonCrest",
    ],
    RecordCategory.TOYS: [
        "PlayBright", "FunForge", "JoyBlock", "ToyVault", "KidCraft",
        "GameGear", "SparkPlay", "MagicMold", "WonderWorks", "BrickBay",
    ],
    RecordCategory.BEAUTY: [
        "GlowLab", "PureBeauty", "SkinSage", "NaturalAura", "BlushCraft",
        "LuminEssence", "VelvetGlow", "ClearSkin", "HerbGlow", "RadiantBase",
    ],
    RecordCategory.AUTOMOTIVE: [
        "AutoEdge", "DriveCore", "CarCraft", "WheelWorks", "MotorMate",
        "SparkDrive", "GrillPro", "TireTech", "GearShift", "LaneGuide",
    ],
    RecordCategory.FOOD: [
        "FarmFresh", "GrainGood", "NutriPack", "PureHarvest", "TasteMakers",
        "GreenBite", "NaturalFarm", "RichBlend", "SpiceTrace", "FreshRoot",
    ],
    RecordCategory.OFFICE: [
        "DeskPro", "PaperEdge", "OfficeSync", "PenCraft", "NoteVault",
        "FileCo", "WorkCore", "DesignDesk", "BriefMate", "TaskMark",
    ],
}

_RATING_WEIGHTS = [0.05, 0.08, 0.12, 0.30, 0.45]  # Ratings 1-5, skewed toward 5

_REVIEW_TEMPLATES = [
    "Great product, works as expected.",
    "Good quality for the price.",
    "Exceeded my expectations!",
    "Decent product, nothing special.",
    "Would not recommend, poor quality.",
    "Amazing value, highly recommend.",
    "Arrived quickly and in perfect condition.",
    "Exactly as described, very satisfied.",
    "A bit disappointing, but functional.",
    "Outstanding build quality and performance.",
    "Works well but packaging was damaged.",
    "Perfect fit for my needs.",
    "Returned due to defects.",
    "Best purchase I've made this year.",
    "Solid product, reasonable price.",
]

_TITLE_TEMPLATES = [
    "{brand} {category_short} Model {model_num}",
    "{brand} Professional {category_short}",
    "{brand} Premium {category_short} Plus",
    "{brand} Essential {category_short}",
    "{brand} Ultra {category_short} Pro",
    "{category_short} by {brand} - {model_num} Series",
    "{brand} {category_short} {year} Edition",
]

_CATEGORY_SHORT = {
    RecordCategory.ELECTRONICS: "Electronics",
    RecordCategory.BOOKS: "Handbook",
    RecordCategory.HOME_KITCHEN: "Kitchenware",
    RecordCategory.SPORTS: "Equipment",
    RecordCategory.CLOTHING: "Apparel",
    RecordCategory.TOYS: "Playset",
    RecordCategory.BEAUTY: "Skincare",
    RecordCategory.AUTOMOTIVE: "Auto Part",
    RecordCategory.FOOD: "Food Product",
    RecordCategory.OFFICE: "Office Supply",
}

# Timestamp distribution: clustered around specific periods
_BASE_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
_DATE_RANGE_DAYS = 1460  # ~4 years


# ---------------------------------------------------------------------------
# WorldManifest
# ---------------------------------------------------------------------------


class WorldManifest(BaseModel):
    """Manifest for a generated corpus world."""

    world_id: str
    generator_version: str = GENERATOR_VERSION
    seed: int
    scale_n: int
    schema_version: str = SCHEMA_VERSION
    row_count: int
    parquet_path: str
    jsonl_path: str
    parquet_hash: str
    jsonl_hash: str
    summary_stats: dict[str, Any]
    parent_world_id: Optional[str] = None
    created_at: str


# ---------------------------------------------------------------------------
# TrackMGenerator
# ---------------------------------------------------------------------------


class TrackMGenerator:
    """
    Deterministic Track M corpus generator.
    
    For a fixed seed and generator version, the output is byte-identical.
    For a fixed seed, records in N=10 are a deterministic subset of N=50,
    which are a subset of N=200, which are a subset of N=1000 (nestedness).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng: Optional[random.Random] = None

    def _make_rng(self) -> random.Random:
        """Create a seeded RNG from seed + generator version for stability."""
        seed_str = f"{GENERATOR_VERSION}:{self.seed}"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big")
        return random.Random(seed_int)

    # Fixed pool size for product IDs - MUST NOT CHANGE for nestedness guarantee
    _PRODUCT_ID_POOL = 2000

    def generate_records(self, n: int) -> list[CorpusRecord]:
        """
        Generate exactly n records deterministically.
        
        Records are generated in a fixed sequence; generating n=50 returns
        the n=10 records as the first 10, guaranteeing nestedness.
        
        NESTEDNESS GUARANTEE: Product IDs are drawn from a fixed pool of
        _PRODUCT_ID_POOL entries generated with a separate RNG before the
        record-generation RNG starts. This ensures the main RNG sequence
        is identical regardless of n.
        """
        # Separate RNG for product ID pool (always fixed size)
        pool_rng = random.Random(f"pool:{GENERATOR_VERSION}:{self.seed}")
        product_ids = [self._gen_asin(pool_rng) for _ in range(self._PRODUCT_ID_POOL)]
        
        rng = self._make_rng()
        records = []
        
        for i in range(n):
            record = self._generate_record(rng, i, product_ids)
            records.append(record)
        
        return records

    def _gen_asin(self, rng: random.Random) -> str:
        """Generate a deterministic ASIN-like product ID."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "B" + "".join(rng.choice(chars) for _ in range(9))

    def _generate_record(
        self, rng: random.Random, index: int, product_ids: list[str]
    ) -> CorpusRecord:
        """Generate a single record with realistic distributions."""
        category = rng.choices(_CATEGORY_NAMES, weights=_CATEGORY_WEIGHTS)[0]
        brands = _BRANDS_BY_CATEGORY[category]
        
        # Long-tail brand distribution: top 3 brands get 60% of reviews
        brand_weights = [0.25, 0.20, 0.15] + [0.4 / (len(brands) - 3)] * (len(brands) - 3)
        brand = rng.choices(brands, weights=brand_weights[:len(brands)])[0]
        
        # Rating with realistic skew toward 4-5 stars
        rating = float(rng.choices([1, 2, 3, 4, 5], weights=_RATING_WEIGHTS)[0])
        
        # Add half-star variants occasionally
        if rng.random() < 0.3:
            rating = min(5.0, rating + 0.5)
        
        # Timestamp with clustering
        base_offset = int(rng.gauss(180, 120))
        base_offset = max(0, min(_DATE_RANGE_DAYS, base_offset))
        # Add clusters around Q4 shopping seasons
        if rng.random() < 0.35:
            # Cluster around November-December
            year_offset = rng.randint(0, 3) * 365
            day_offset = rng.randint(305, 365)  # Nov-Dec
            base_offset = year_offset + day_offset
            base_offset = min(_DATE_RANGE_DAYS, base_offset)
        event_time = _BASE_DATE + timedelta(days=base_offset)
        
        # Price: sparse (30% missing), realistic distribution
        price = None
        if rng.random() > 0.30:
            price = Decimal(str(round(rng.lognormvariate(3.5, 1.2), 2)))
            price = max(Decimal("0.99"), min(Decimal("999.99"), price))
        
        # Helpful votes: sparse, heavy-tail
        helpful_votes = 0
        if rng.random() < 0.35:
            helpful_votes = int(rng.paretovariate(1.5))
            helpful_votes = min(helpful_votes, 10000)
        
        # Verified purchase: 70% verified
        verified_purchase = rng.random() < 0.70
        
        # Title: may have duplicates across products
        title_tmpl = rng.choice(_TITLE_TEMPLATES)
        model_num = rng.randint(100, 999)
        year = rng.choice([2020, 2021, 2022, 2023])
        title = title_tmpl.format(
            brand=brand,
            category_short=_CATEGORY_SHORT[category],
            model_num=model_num,
            year=year,
        )
        
        # Product ID: use pre-generated IDs, with some parent relationships
        product_id = product_ids[index % len(product_ids)]
        parent_id = None
        if rng.random() < 0.25 and index >= 5:
            parent_id = product_ids[(index - rng.randint(1, 5)) % len(product_ids)]
        
        # Source record ID
        source_record_id = f"R{self.seed:04d}_{index:06d}"
        
        # World ID placeholder (will be set by caller)
        world_id = f"seed_{self.seed}"
        
        # Raw payload hash
        raw_payload = {
            "index": index,
            "seed": self.seed,
            "generator_version": GENERATOR_VERSION,
        }
        raw_payload_hash = hashlib.sha256(
            orjson.dumps(raw_payload, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()[:32]
        
        # Review text
        text = rng.choice(_REVIEW_TEMPLATES)
        
        # Deterministic record ID
        record_id = f"rec_{self.seed:04d}_{index:06d}"
        
        # Attributes: extensible
        attributes: dict[str, Any] = {
            "subcategory": f"{category.value}/{brand}",
        }
        if rng.random() < 0.4:
            attributes["color"] = rng.choice(["Black", "White", "Red", "Blue", "Green"])
        if rng.random() < 0.3:
            attributes["size"] = rng.choice(["S", "M", "L", "XL", "One Size"])
        
        return CorpusRecord(
            record_id=record_id,
            source="track_m_synthetic",
            source_record_id=source_record_id,
            world_id=world_id,
            product_id=product_id,
            parent_id=parent_id,
            category=category,
            title=title,
            brand=brand,
            rating=rating,
            helpful_votes=helpful_votes,
            verified_purchase=verified_purchase,
            event_time=event_time,
            price=price,
            attributes=attributes,
            text=text,
            raw_payload_hash=raw_payload_hash,
        )

    def generate_world(
        self,
        n: int,
        output_dir: Path,
        world_id: Optional[str] = None,
        parent_world_id: Optional[str] = None,
    ) -> tuple[CorpusWorld, WorldManifest]:
        """
        Generate a corpus world of n records and save to disk.
        
        Returns (CorpusWorld, WorldManifest).
        """
        if world_id is None:
            world_id = f"world_s{self.seed}_n{n}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate records
        records = self.generate_records(n)
        
        # Update world_id in records
        updated_records = []
        for r in records:
            updated_records.append(r.model_copy(update={"world_id": world_id}))
        records = updated_records
        
        # Convert to DataFrame
        df = self._records_to_df(records)
        
        # Save Parquet
        parquet_path = output_dir / "records.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, parquet_path, compression="snappy")
        
        # Save JSONL
        jsonl_path = output_dir / "records.jsonl"
        with open(jsonl_path, "wb") as f:
            for record in records:
                f.write(record.model_dump_json_bytes())
                f.write(b"\n")
        
        # Compute file hashes
        parquet_hash = _file_hash(parquet_path)
        jsonl_hash = _file_hash(jsonl_path)
        
        # Summary statistics
        summary_stats = self._compute_summary(df)
        
        # Record IDs hash
        sorted_ids = sorted(r.record_id for r in records)
        record_ids_hash = hashlib.sha256(
            "|".join(sorted_ids).encode()
        ).hexdigest()[:32]
        
        # Create CorpusWorld
        world = CorpusWorld(
            world_id=world_id,
            seed=self.seed,
            scale_n=n,
            parent_world_id=parent_world_id,
            record_ids_hash=record_ids_hash,
            manifest_path=str(output_dir / "manifest.json"),
        )
        
        # Create manifest
        now = datetime.now(timezone.utc).isoformat()
        manifest = WorldManifest(
            world_id=world_id,
            seed=self.seed,
            scale_n=n,
            row_count=len(records),
            parquet_path=str(parquet_path),
            jsonl_path=str(jsonl_path),
            parquet_hash=parquet_hash,
            jsonl_hash=jsonl_hash,
            summary_stats=summary_stats,
            parent_world_id=parent_world_id,
            created_at=now,
        )
        
        # Save manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(), indent=2, default=str)
        )
        
        return world, manifest

    def generate_nested_worlds(
        self,
        scales: list[int],
        output_dir: Path,
    ) -> list[tuple[CorpusWorld, WorldManifest]]:
        """
        Generate a sequence of nested worlds at increasing scales.
        
        Scales must be sorted ascending. Each larger world is a superset of
        the smaller world (nestedness guarantee via fixed RNG sequence).
        """
        scales_sorted = sorted(scales)
        results = []
        parent_world_id = None
        
        for n in scales_sorted:
            world_id = f"world_s{self.seed}_n{n}"
            world_dir = output_dir / world_id
            world, manifest = self.generate_world(
                n=n,
                output_dir=world_dir,
                world_id=world_id,
                parent_world_id=parent_world_id,
            )
            results.append((world, manifest))
            parent_world_id = world_id
        
        return results

    def generate_adversarial_fixtures(self, output_dir: Path) -> dict[str, Any]:
        """
        Generate controlled adversarial edge cases.
        
        These are separate from the main corpus worlds and used for
        testing boundary conditions in the gold engine.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        fixtures: dict[str, Any] = {}
        rng = random.Random(f"adversarial:{self.seed}")
        
        # 1. Tie in ranking: multiple brands with identical counts
        tie_records = self._gen_tie_fixture(rng)
        fixtures["ties"] = tie_records
        
        # 2. Empty scope: predicate that matches no records
        empty_record = self._gen_base_record(rng, 0, "empty_world")
        fixtures["empty_scope_sample"] = [empty_record]
        
        # 3. One-record scope
        one_record = self._gen_base_record(rng, 1, "one_record_world")
        fixtures["one_record"] = [one_record]
        
        # 4. Null-heavy: 90% of prices missing
        null_records = []
        for i in range(20):
            r = self._gen_base_record(rng, i, "null_heavy_world")
            if rng.random() < 0.9:
                r = r.model_copy(update={"price": None})
            null_records.append(r)
        fixtures["null_heavy"] = null_records
        
        # 5. Boundary dates: records at exact date boundaries
        boundary_records = []
        boundary_times = [
            datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2022, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        ]
        for i, bt in enumerate(boundary_times):
            r = self._gen_base_record(rng, i, "boundary_world")
            boundary_records.append(r.model_copy(update={"event_time": bt}))
        fixtures["boundary_dates"] = boundary_records
        
        # 6. Near-equal means: two groups with means differing by < 0.01
        near_equal = []
        for i in range(10):
            r = self._gen_base_record(rng, i, "near_equal_world")
            rating = 3.5 + (i % 2) * 0.005  # 3.5 vs 3.505
            near_equal.append(r.model_copy(update={"rating": rating}))
        fixtures["near_equal_means"] = near_equal
        
        # Save fixtures
        for name, fixture_records in fixtures.items():
            if isinstance(fixture_records, list) and len(fixture_records) > 0:
                jsonl_path = output_dir / f"{name}.jsonl"
                with open(jsonl_path, "wb") as f:
                    for r in fixture_records:
                        f.write(r.model_dump_json_bytes())
                        f.write(b"\n")
        
        return {k: len(v) if isinstance(v, list) else v for k, v in fixtures.items()}

    def _gen_tie_fixture(self, rng: random.Random) -> list[CorpusRecord]:
        """Generate records where multiple brands have identical review counts."""
        records = []
        tied_brands = ["BrandAlpha", "BrandBeta", "BrandGamma"]
        for brand in tied_brands:
            for i in range(5):  # exactly 5 records per brand
                r = self._gen_base_record(rng, len(records), "tie_world")
                records.append(r.model_copy(update={"brand": brand}))
        return records

    def _gen_base_record(self, rng: random.Random, index: int, world_id: str) -> CorpusRecord:
        """Generate a minimal valid record for fixture use."""
        return CorpusRecord(
            record_id=f"fix_{world_id}_{index:04d}",
            source_record_id=f"F_{index:06d}",
            world_id=world_id,
            product_id=f"BFIX{index:06d}",
            category=RecordCategory.ELECTRONICS,
            title=f"Fixture Product {index}",
            brand=f"FixtureBrand{index % 3}",
            rating=float(rng.choices([1, 2, 3, 4, 5], weights=_RATING_WEIGHTS)[0]),
            verified_purchase=True,
            event_time=_BASE_DATE + timedelta(days=index * 30),
            raw_payload_hash=hashlib.sha256(f"fix:{index}".encode()).hexdigest()[:32],
        )

    def _records_to_df(self, records: list[CorpusRecord]) -> pd.DataFrame:
        """Convert records to a Pandas DataFrame suitable for Parquet storage."""
        rows = []
        for r in records:
            row = {
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
                "text": r.text,
                "raw_payload_hash": r.raw_payload_hash,
                "schema_version": r.schema_version,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _compute_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute summary statistics for the manifest."""
        stats: dict[str, Any] = {
            "total_records": len(df),
            "category_counts": df["category"].value_counts().to_dict(),
            "brand_count": int(df["brand"].nunique()),
            "rating_mean": float(df["rating"].mean()),
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
            "verified_purchase_ratio": float(df["verified_purchase"].mean()),
            "price_missing_ratio": float(df["price"].isna().mean()),
            "helpful_votes_mean": float(df["helpful_votes"].mean()),
            "date_range": {
                "min": str(df["event_time"].min()),
                "max": str(df["event_time"].max()),
            },
        }
        return stats


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:32]
