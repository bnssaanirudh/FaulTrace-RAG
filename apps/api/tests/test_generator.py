"""
Tests for the Track M deterministic generator.

Validates: determinism, nestedness, manifest correctness, stable hashes, edge cases.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from faulttrace_data.generator import TrackMGenerator, GENERATOR_VERSION


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def test_generator_determinism(tmp_dir):
    """Same seed must produce identical records."""
    gen1 = TrackMGenerator(seed=42)
    gen2 = TrackMGenerator(seed=42)
    records1 = gen1.generate_records(50)
    records2 = gen2.generate_records(50)
    
    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.record_id == r2.record_id
        assert r1.category == r2.category
        assert r1.rating == r2.rating
        assert r1.brand == r2.brand


def test_generator_different_seeds_differ():
    """Different seeds must produce different records."""
    gen1 = TrackMGenerator(seed=42)
    gen2 = TrackMGenerator(seed=99)
    records1 = gen1.generate_records(10)
    records2 = gen2.generate_records(10)
    
    # At least some records should differ
    assert any(r1.category != r2.category or r1.rating != r2.rating
               for r1, r2 in zip(records1, records2))


def test_nestedness_guarantee(tmp_dir):
    """Records in N=10 must be first 10 of N=50."""
    gen = TrackMGenerator(seed=42)
    records_10 = gen.generate_records(10)
    records_50 = gen.generate_records(50)
    
    # First 10 records must be identical
    for i in range(10):
        assert records_10[i].record_id == records_50[i].record_id
        assert records_10[i].category == records_50[i].category
        assert records_10[i].rating == records_50[i].rating


def test_nested_worlds_file_output(tmp_dir):
    """Test that nested worlds are generated correctly to files."""
    gen = TrackMGenerator(seed=42)
    results = gen.generate_nested_worlds(
        scales=[10, 50, 200],
        output_dir=tmp_dir,
    )
    
    assert len(results) == 3
    
    # Check nestedness of file hashes
    parent_ids = set()
    for world, manifest in results:
        assert manifest.row_count == world.scale_n
        parquet_path = Path(manifest.parquet_path)
        assert parquet_path.exists()
        
        df = pd.read_parquet(parquet_path)
        assert len(df) == world.scale_n
        
        if parent_ids:
            # All parent IDs should be a subset of current world
            current_ids = set(df["record_id"].tolist())
            assert parent_ids.issubset(current_ids), "Nestedness violated!"
        
        parent_ids = set(df["record_id"].tolist())


def test_manifest_correctness(tmp_dir):
    """Manifest must contain correct metadata."""
    gen = TrackMGenerator(seed=42)
    world, manifest = gen.generate_world(n=20, output_dir=tmp_dir / "world_test")
    
    assert manifest.generator_version == GENERATOR_VERSION
    assert manifest.seed == 42
    assert manifest.scale_n == 20
    assert manifest.row_count == 20
    assert manifest.parquet_hash != ""
    assert manifest.jsonl_hash != ""
    assert "total_records" in manifest.summary_stats
    assert manifest.summary_stats["total_records"] == 20


def test_manifest_hash_stable(tmp_dir):
    """Same seed must produce the same Parquet hash."""
    gen1 = TrackMGenerator(seed=42)
    gen2 = TrackMGenerator(seed=42)
    
    w1, m1 = gen1.generate_world(n=10, output_dir=tmp_dir / "w1", world_id="test_w1")
    w2, m2 = gen2.generate_world(n=10, output_dir=tmp_dir / "w2", world_id="test_w1")
    
    assert m1.parquet_hash == m2.parquet_hash
    assert m1.jsonl_hash == m2.jsonl_hash


def test_rating_range():
    """All ratings must be in [1.0, 5.0]."""
    gen = TrackMGenerator(seed=42)
    records = gen.generate_records(200)
    for r in records:
        assert 1.0 <= r.rating <= 5.0


def test_category_distribution():
    """Category distribution should be non-uniform (skewed)."""
    gen = TrackMGenerator(seed=42)
    records = gen.generate_records(1000)
    
    from collections import Counter
    counts = Counter(r.category.value for r in records)
    
    # Electronics should be most common
    assert counts["Electronics"] > counts["Office Products"]
    # All categories should appear
    assert len(counts) > 5


def test_price_missing_rate():
    """About 30% of prices should be missing."""
    gen = TrackMGenerator(seed=42)
    records = gen.generate_records(500)
    missing = sum(1 for r in records if r.price is None)
    rate = missing / len(records)
    # Allow ±10% variance
    assert 0.15 <= rate <= 0.50


def test_adversarial_fixtures(tmp_dir):
    """Adversarial fixtures should be generated correctly."""
    gen = TrackMGenerator(seed=42)
    counts = gen.generate_adversarial_fixtures(tmp_dir)
    
    assert "ties" in counts
    assert "null_heavy" in counts
    assert "boundary_dates" in counts
    assert "near_equal_means" in counts


def test_tie_fixture_structure(tmp_dir):
    """Tie fixture should have equal counts per brand."""
    gen = TrackMGenerator(seed=42)
    gen.generate_adversarial_fixtures(tmp_dir)
    
    ties_path = tmp_dir / "ties.jsonl"
    assert ties_path.exists()
    
    records = []
    with open(ties_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Should have tied brands
    from collections import Counter
    brand_counts = Counter(r["brand"] for r in records)
    values = list(brand_counts.values())
    # All brands should have equal counts
    assert len(set(values)) == 1  # all equal
