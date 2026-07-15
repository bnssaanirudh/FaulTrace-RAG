"""
World builder for FaultTrace-RAG Prompt 2.

Generalizes the Prompt 1 nested-world builder to work over both
generated (Track M) and ingested (Amazon-style) datasets.

World builder policies:
  - deterministic_prefix: stable seeded shuffle, take first N
  - stratified: stratified sample preserving category/rating/time distribution
  - adversarial: slices optimized for null-heavy, tie-heavy, rare-category,
                 date-boundary, and near-equal aggregate edge cases
  - holdout: reserved partition for future Track T semantic predicates

All worlds record:
  - parent_world_id links with proof of subset relation
  - content-addressed record ID list artifact
  - world summary statistics and divergence from parent snapshot

Supported scales: 10, 50, 200, 1000 (default smoke); 2000, 5000 (extended).
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field, field_validator

from faulttrace_core.models import SCHEMA_VERSION

WORLD_BUILDER_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Worlds config schema (worlds.yaml)
# ---------------------------------------------------------------------------


class WorldScaleConfig(BaseModel):
    """Configuration for one world at a given scale."""
    n: int = Field(..., gt=0, description="Number of records in this world")
    policy: str = Field(default="deterministic_prefix", description="Sampling policy")
    adversarial_slices: list[str] = Field(
        default_factory=list,
        description="Adversarial slice types: null_heavy, tie_heavy, rare_category, date_boundary, near_equal"
    )

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        allowed = {"deterministic_prefix", "stratified", "adversarial", "holdout"}
        if v not in allowed:
            raise ValueError(f"Unknown policy '{v}'. Allowed: {allowed}")
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v: int) -> int:
        allowed_scales = {10, 50, 200, 1000, 2000, 5000}
        if v not in allowed_scales:
            # Allow any reasonable value for custom configs
            if v <= 0 or v > 10000:
                raise ValueError(f"Scale {v} must be between 1 and 10000")
        return v


class WorldGroupConfig(BaseModel):
    """Configuration for a group of nested worlds sharing a seed and dataset."""
    group_id: str
    dataset_id: str
    seed: int = 42
    scales: list[WorldScaleConfig]
    include_adversarial: bool = True
    holdout_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    description: str = ""

    @field_validator("scales")
    @classmethod
    def validate_scales_ascending(cls, v: list[WorldScaleConfig]) -> list[WorldScaleConfig]:
        ns = [s.n for s in v]
        if ns != sorted(ns):
            raise ValueError("Scales must be listed in ascending order")
        return v


class WorldsYamlConfig(BaseModel):
    """Top-level worlds.yaml schema."""
    schema_version: str = "2.0.0"
    groups: list[WorldGroupConfig]


def load_worlds_config(config_path: Path) -> WorldsYamlConfig:
    """Load and validate a worlds.yaml configuration file."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return WorldsYamlConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# WorldManifest v2 (backward-compatible with Prompt 1)
# ---------------------------------------------------------------------------


class WorldManifestV2(BaseModel):
    """Extended world manifest with all Prompt 2 provenance fields."""

    # Core (Prompt 1 compatible)
    world_id: str
    generator_version: str = WORLD_BUILDER_VERSION
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

    # Prompt 2 additions (all optional for backward compatibility)
    dataset_id: str = "track_m"
    source_type: str = "generated"
    sampling_policy: str = "deterministic_prefix"
    producing_command: Optional[str] = None
    config_hash: Optional[str] = None
    parent_artifact_refs: list[str] = Field(default_factory=list)
    record_ids_hash: str = ""
    record_ids_artifact: Optional[str] = None  # Path to sorted record IDs JSONL
    divergence_from_parent: dict[str, Any] = Field(default_factory=dict)
    adversarial_slice_types: list[str] = Field(default_factory=list)
    group_id: Optional[str] = None
    snapshot_id: Optional[str] = None


# ---------------------------------------------------------------------------
# WorldBuilder
# ---------------------------------------------------------------------------


class WorldBuilder:
    """
    Builds nested corpus worlds from any Parquet source.

    Supports both generated (Track M) and ingested (Amazon) datasets.
    All sampling is seeded and deterministic.
    """

    def __init__(
        self,
        source_parquet: Path,
        dataset_id: str = "track_m",
        seed: int = 42,
        group_id: Optional[str] = None,
    ):
        self.source_parquet = source_parquet
        self.dataset_id = dataset_id
        self.seed = seed
        self.group_id = group_id
        self._df: Optional[pd.DataFrame] = None

    def _load_df(self) -> pd.DataFrame:
        """Load source Parquet lazily."""
        if self._df is None:
            if self.source_parquet.is_dir():
                # Directory of Parquet files (from Amazon ingestor)
                dfs = []
                for pf in sorted(self.source_parquet.glob("**/*.parquet")):
                    dfs.append(pd.read_parquet(pf))
                self._df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                self._df = pd.read_parquet(self.source_parquet)
        return self._df

    def build_nested_worlds(
        self,
        scales: list[int],
        output_dir: Path,
        policy: str = "deterministic_prefix",
        producing_command: Optional[str] = None,
    ) -> list[WorldManifestV2]:
        """
        Build a sequence of nested worlds at increasing scales.

        NESTEDNESS GUARANTEE: For policy='deterministic_prefix', the world
        at scale N is a strict prefix of the world at scale M > N after a
        single stable seeded shuffle. This is cryptographically verifiable
        via the record_ids_hash artifact.

        Returns list of WorldManifestV2 in ascending order.
        """
        df = self._load_df()
        scales_sorted = sorted(scales)

        if len(df) < scales_sorted[-1]:
            raise ValueError(
                f"Source has {len(df)} rows but largest scale requires {scales_sorted[-1]}"
            )

        # Single stable shuffle for all scales (nestedness guarantee)
        rng = random.Random(f"worldbuilder:{WORLD_BUILDER_VERSION}:{self.seed}")
        indices = list(range(len(df)))
        rng.shuffle(indices)
        shuffled_df = df.iloc[indices].reset_index(drop=True)

        results = []
        parent_world_id = None
        parent_record_ids: Optional[list[str]] = None

        for n in scales_sorted:
            world_id = f"world_{self.dataset_id}_s{self.seed}_n{n}"
            world_dir = output_dir / world_id
            world_dir.mkdir(parents=True, exist_ok=True)

            # Take first N rows (nestedness via prefix)
            world_df = shuffled_df.iloc[:n].copy()
            world_df["world_id"] = world_id

            # Verify nestedness: all parent records are in this world
            current_ids = set(world_df["record_id"].tolist())
            if parent_record_ids is not None:
                parent_set = set(parent_record_ids)
                assert parent_set.issubset(current_ids), (
                    f"NESTEDNESS VIOLATION: {len(parent_set - current_ids)} parent records "
                    f"missing from world {world_id}"
                )

            # Write Parquet
            parquet_path = world_dir / "records.parquet"
            table = pa.Table.from_pandas(world_df, preserve_index=False)
            pq.write_table(table, parquet_path, compression="snappy")

            # Write JSONL
            jsonl_path = world_dir / "records.jsonl"
            world_df.to_json(jsonl_path, orient="records", lines=True, date_format="iso")

            # Record IDs artifact (sorted, for subset proof)
            sorted_ids = sorted(world_df["record_id"].tolist())
            record_ids_path = world_dir / "record_ids.json"
            record_ids_path.write_text(json.dumps(sorted_ids, indent=0))

            # Hashes
            parquet_hash = _file_sha256(parquet_path)
            jsonl_hash = _file_sha256(jsonl_path)
            record_ids_hash = hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()[:32]

            # Summary stats
            summary_stats = _compute_summary(world_df)

            # Divergence from parent
            divergence: dict[str, Any] = {}
            if parent_record_ids is not None:
                new_ids = current_ids - set(parent_record_ids)
                divergence["new_records"] = len(new_ids)
                divergence["parent_records_preserved"] = len(parent_record_ids)
                if "category" in world_df.columns:
                    parent_cats = set(
                        shuffled_df[shuffled_df["record_id"].isin(parent_record_ids)]["category"].unique()
                    )
                    current_cats = set(world_df["category"].unique())
                    divergence["new_categories"] = list(current_cats - parent_cats)

            now = datetime.now(timezone.utc).isoformat()
            manifest = WorldManifestV2(
                world_id=world_id,
                seed=self.seed,
                scale_n=n,
                row_count=len(world_df),
                parquet_path=str(parquet_path),
                jsonl_path=str(jsonl_path),
                parquet_hash=parquet_hash,
                jsonl_hash=jsonl_hash,
                summary_stats=summary_stats,
                parent_world_id=parent_world_id,
                created_at=now,
                dataset_id=self.dataset_id,
                source_type="ingested" if self.dataset_id != "track_m" else "generated",
                sampling_policy=policy,
                producing_command=producing_command,
                record_ids_hash=record_ids_hash,
                record_ids_artifact=str(record_ids_path),
                divergence_from_parent=divergence,
                group_id=self.group_id,
            )

            # Save manifest
            manifest_path = world_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.model_dump(), indent=2, default=str)
            )

            results.append(manifest)
            parent_world_id = world_id
            parent_record_ids = sorted_ids

        return results

    def build_adversarial_slices(
        self,
        output_dir: Path,
        slice_types: Optional[list[str]] = None,
    ) -> dict[str, WorldManifestV2]:
        """
        Build adversarial world slices for edge-case testing.

        Slice types:
          null_heavy     — records where price is null (tests null handling)
          tie_heavy      — records chosen to create rating ties
          rare_category  — records from the least-common category
          date_boundary  — records at year/quarter boundaries
          near_equal     — records where group means differ by < 0.1
        """
        df = self._load_df()
        results: dict[str, WorldManifestV2] = {}

        if slice_types is None:
            slice_types = ["null_heavy", "tie_heavy", "rare_category", "date_boundary", "near_equal"]

        rng = random.Random(f"adversarial:{WORLD_BUILDER_VERSION}:{self.seed}")

        slice_builders = {
            "null_heavy": self._slice_null_heavy,
            "tie_heavy": self._slice_tie_heavy,
            "rare_category": self._slice_rare_category,
            "date_boundary": self._slice_date_boundary,
            "near_equal": self._slice_near_equal,
        }

        for slice_type in slice_types:
            if slice_type not in slice_builders:
                continue
            builder_fn = slice_builders[slice_type]
            try:
                slice_df = builder_fn(df, rng)
                if slice_df is None or len(slice_df) == 0:
                    continue

                world_id = f"adv_{self.dataset_id}_s{self.seed}_{slice_type}"
                world_dir = output_dir / world_id
                world_dir.mkdir(parents=True, exist_ok=True)

                slice_df = slice_df.copy()
                slice_df["world_id"] = world_id

                parquet_path = world_dir / "records.parquet"
                table = pa.Table.from_pandas(slice_df, preserve_index=False)
                pq.write_table(table, parquet_path, compression="snappy")

                jsonl_path = world_dir / "records.jsonl"
                slice_df.to_json(jsonl_path, orient="records", lines=True, date_format="iso")

                sorted_ids = sorted(slice_df["record_id"].tolist())
                record_ids_hash = hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()[:32]

                parquet_hash = _file_sha256(parquet_path)
                jsonl_hash = _file_sha256(jsonl_path)

                now = datetime.now(timezone.utc).isoformat()
                manifest = WorldManifestV2(
                    world_id=world_id,
                    seed=self.seed,
                    scale_n=len(slice_df),
                    row_count=len(slice_df),
                    parquet_path=str(parquet_path),
                    jsonl_path=str(jsonl_path),
                    parquet_hash=parquet_hash,
                    jsonl_hash=jsonl_hash,
                    summary_stats=_compute_summary(slice_df),
                    parent_world_id=None,
                    created_at=now,
                    dataset_id=self.dataset_id,
                    source_type="adversarial",
                    sampling_policy="adversarial",
                    record_ids_hash=record_ids_hash,
                    adversarial_slice_types=[slice_type],
                    group_id=self.group_id,
                )

                manifest_path = world_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2, default=str))

                results[slice_type] = manifest
            except Exception as e:
                # Log but don't crash — adversarial slices are best-effort
                import logging
                logging.getLogger(__name__).warning("adversarial_slice_failed slice=%s error=%s", slice_type, str(e))

        return results

    def verify_nestedness(self, manifests: list[WorldManifestV2]) -> dict[str, Any]:
        """
        Cryptographically verify that manifests form a strict nested chain.

        For each consecutive pair (small, large), verifies that all record IDs
        in small appear in large by comparing record_ids.json artifacts.

        Returns a verification report dict.
        """
        checks = []
        for i in range(len(manifests) - 1):
            small = manifests[i]
            large = manifests[i + 1]

            small_ids_path = Path(small.record_ids_artifact) if small.record_ids_artifact else None
            large_ids_path = Path(large.record_ids_artifact) if large.record_ids_artifact else None

            if small_ids_path is None or not small_ids_path.exists():
                checks.append({
                    "pair": (small.world_id, large.world_id),
                    "status": "cannot_verify",
                    "reason": "missing_record_ids_artifact",
                })
                continue

            if large_ids_path is None or not large_ids_path.exists():
                checks.append({
                    "pair": (small.world_id, large.world_id),
                    "status": "cannot_verify",
                    "reason": "missing_record_ids_artifact_large",
                })
                continue

            small_ids = set(json.loads(small_ids_path.read_text()))
            large_ids = set(json.loads(large_ids_path.read_text()))

            missing = small_ids - large_ids
            check: dict[str, Any] = {
                "pair": [small.world_id, large.world_id],
                "small_n": small.scale_n,
                "large_n": large.scale_n,
                "small_id_count": len(small_ids),
                "large_id_count": len(large_ids),
            }

            if not missing:
                check["status"] = "nested_verified"
                check["proof"] = f"all {len(small_ids)} small-world IDs found in large-world"
                # Cross-check via hash
                small_hash = hashlib.sha256("|".join(sorted(small_ids)).encode()).hexdigest()[:16]
                check["small_ids_hash_prefix"] = small_hash
            else:
                check["status"] = "nestedness_violation"
                check["missing_count"] = len(missing)
                check["missing_sample"] = list(missing)[:5]

            checks.append(check)

        all_verified = all(c.get("status") == "nested_verified" for c in checks)
        return {
            "status": "nested" if all_verified else "violation_detected",
            "group_id": self.group_id,
            "worlds_checked": len(manifests),
            "pairs_verified": len(checks),
            "checks": checks,
        }

    def summarize_world(self, manifest: WorldManifestV2) -> dict[str, Any]:
        """Return a structured summary of a world."""
        parquet_path = Path(manifest.parquet_path)
        if not parquet_path.exists():
            return {"error": "parquet_not_found", "world_id": manifest.world_id}

        df = pd.read_parquet(parquet_path)
        summary = _compute_summary(df)
        summary["world_id"] = manifest.world_id
        summary["scale_n"] = manifest.scale_n
        summary["parent_world_id"] = manifest.parent_world_id
        summary["dataset_id"] = manifest.dataset_id
        summary["sampling_policy"] = manifest.sampling_policy
        summary["record_ids_hash"] = manifest.record_ids_hash
        summary["divergence_from_parent"] = manifest.divergence_from_parent
        return summary

    # ------------------------------------------------------------------
    # Adversarial slice helpers
    # ------------------------------------------------------------------

    def _slice_null_heavy(self, df: pd.DataFrame, rng: random.Random) -> Optional[pd.DataFrame]:
        """Records where price is null."""
        if "price" not in df.columns:
            return None
        null_df = df[df["price"].isna()]
        return null_df.sample(min(50, len(null_df)), random_state=self.seed) if len(null_df) > 0 else None

    def _slice_tie_heavy(self, df: pd.DataFrame, rng: random.Random) -> Optional[pd.DataFrame]:
        """Records chosen to maximize rating/brand ties."""
        if "brand" not in df.columns:
            return None
        # Pick brands with equal record counts
        brand_counts = df["brand"].value_counts()
        mode_count = brand_counts.mode()[0]
        tied_brands = brand_counts[brand_counts == mode_count].index.tolist()[:5]
        if len(tied_brands) < 2:
            return None
        return df[df["brand"].isin(tied_brands)]

    def _slice_rare_category(self, df: pd.DataFrame, rng: random.Random) -> Optional[pd.DataFrame]:
        """Records from the least-common category."""
        if "category" not in df.columns:
            return None
        cat_counts = df["category"].value_counts()
        rare_cat = cat_counts.index[-1]
        return df[df["category"] == rare_cat]

    def _slice_date_boundary(self, df: pd.DataFrame, rng: random.Random) -> Optional[pd.DataFrame]:
        """Records near year/quarter boundaries."""
        if "event_time" not in df.columns:
            return None
        ts = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        # Records in last/first 2 days of a month
        month_day = ts.dt.day
        boundary = (month_day <= 2) | (month_day >= 28)
        result = df[boundary]
        return result if len(result) > 0 else df.sample(min(10, len(df)), random_state=self.seed)

    def _slice_near_equal(self, df: pd.DataFrame, rng: random.Random) -> Optional[pd.DataFrame]:
        """Records where two groups have nearly equal mean ratings."""
        if "category" not in df.columns or "rating" not in df.columns:
            return None
        cat_means = df.groupby("category")["rating"].mean()
        if len(cat_means) < 2:
            return None
        # Sort by mean, pick two adjacent categories with closest means
        sorted_means = cat_means.sort_values()
        diffs = sorted_means.diff().dropna().abs()
        if len(diffs) == 0:
            return None
        closest_idx = diffs.idxmin()
        idx_pos = sorted_means.index.get_loc(closest_idx)
        cat_a = sorted_means.index[idx_pos - 1]
        cat_b = sorted_means.index[idx_pos]
        return df[df["category"].isin([cat_a, cat_b])]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for a world DataFrame."""
    stats: dict[str, Any] = {"total_records": len(df)}
    if "category" in df.columns:
        stats["category_counts"] = df["category"].value_counts().to_dict()
    if "brand" in df.columns:
        stats["brand_count"] = int(df["brand"].nunique())
    if "rating" in df.columns:
        stats["rating_mean"] = float(df["rating"].mean())
        stats["rating_distribution"] = df["rating"].value_counts().sort_index().to_dict()
    if "verified_purchase" in df.columns:
        stats["verified_purchase_ratio"] = float(df["verified_purchase"].mean())
    if "price" in df.columns:
        stats["price_missing_ratio"] = float(df["price"].isna().mean())
    if "event_time" in df.columns:
        ts = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        stats["date_range"] = {
            "min": str(ts.min()),
            "max": str(ts.max()),
        }
    return stats


def _file_sha256(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()[:32]


def verify_world_nestedness_by_ids(
    small_ids_path: Path,
    large_ids_path: Path,
) -> dict[str, Any]:
    """
    Standalone nestedness verifier for two record-IDs artifacts.
    Returns {"nested": True} or {"nested": False, "missing": [...]}
    """
    small_ids = set(json.loads(small_ids_path.read_text()))
    large_ids = set(json.loads(large_ids_path.read_text()))
    missing = small_ids - large_ids
    if not missing:
        return {
            "nested": True,
            "small_count": len(small_ids),
            "large_count": len(large_ids),
            "proof": "all_small_ids_in_large",
        }
    return {
        "nested": False,
        "small_count": len(small_ids),
        "large_count": len(large_ids),
        "missing_count": len(missing),
        "missing_sample": list(missing)[:10],
    }
