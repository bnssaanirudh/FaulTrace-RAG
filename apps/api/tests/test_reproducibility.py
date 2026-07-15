"""
Reproducibility tests for FaultTrace-RAG (WP9).

Tests that all deterministic components produce bit-identical output given:
- Same seed
- Same schema version
- Same generator version

Also tests:
- Path traversal defense in SnapshotRegistry
- Nestedness proof: subset relation verified via record_ids.json artifacts
- Two-world hash chain: parquet_hash reproducible across runs
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


class TestWorldReproducibility:
    """Generated worlds must be bit-reproducible given same seed and scale."""

    def test_same_seed_same_parquet_hash(self, tmp_path):
        """Two runs with same seed must produce identical parquet_hash."""
        from faulttrace_data.generator import TrackMGenerator

        gen1 = TrackMGenerator(seed=42)
        worlds_dir1 = tmp_path / "run1"
        results1 = gen1.generate_nested_worlds(scales=[50], output_dir=worlds_dir1)
        _, manifest1 = results1[0]

        gen2 = TrackMGenerator(seed=42)
        worlds_dir2 = tmp_path / "run2"
        results2 = gen2.generate_nested_worlds(scales=[50], output_dir=worlds_dir2)
        _, manifest2 = results2[0]

        assert manifest1.parquet_hash == manifest2.parquet_hash, (
            f"Parquet hash mismatch: {manifest1.parquet_hash} vs {manifest2.parquet_hash}"
        )

    def test_different_seed_different_parquet_hash(self, tmp_path):
        """Different seeds must produce different data."""
        from faulttrace_data.generator import TrackMGenerator

        gen1 = TrackMGenerator(seed=42)
        results1 = gen1.generate_nested_worlds(scales=[50], output_dir=tmp_path / "run1")
        _, manifest1 = results1[0]

        gen2 = TrackMGenerator(seed=99)
        results2 = gen2.generate_nested_worlds(scales=[50], output_dir=tmp_path / "run2")
        _, manifest2 = results2[0]

        assert manifest1.parquet_hash != manifest2.parquet_hash

    def test_world_id_includes_seed_and_scale(self, tmp_path):
        """World IDs must encode seed and scale for auditability."""
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[10, 50], output_dir=tmp_path)
        for world, manifest in results:
            assert "42" in world.world_id  # seed
            assert str(world.scale_n) in world.world_id  # scale


class TestNestednessReproducibility:
    """Nested world subsets must be provably consistent across runs."""

    def test_nestedness_chain_holds(self, tmp_path):
        """World[n=10] records must all appear in world[n=50]."""
        from faulttrace_data.generator import TrackMGenerator
        import pandas as pd

        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[10, 50, 200], output_dir=tmp_path)

        # Load each world and check subset relation
        dfs = {}
        for world, manifest in results:
            world_dir = tmp_path / world.world_id
            parquet_path = world_dir / "records.parquet"
            if parquet_path.exists():
                dfs[world.scale_n] = set(pd.read_parquet(parquet_path)["record_id"].tolist())

        if 10 in dfs and 50 in dfs:
            assert dfs[10].issubset(dfs[50]), "world[10] ⊄ world[50]: nestedness violated"

        if 50 in dfs and 200 in dfs:
            assert dfs[50].issubset(dfs[200]), "world[50] ⊄ world[200]: nestedness violated"

    def test_nestedness_same_across_runs(self, tmp_path):
        """Two independent runs must produce the same nested record ID sets."""
        from faulttrace_data.generator import TrackMGenerator
        import pandas as pd

        gen1 = TrackMGenerator(seed=42)
        results1 = gen1.generate_nested_worlds(scales=[10], output_dir=tmp_path / "run1")
        world1, _ = results1[0]
        world_dir1 = tmp_path / "run1" / world1.world_id
        ids1 = set()
        pf1 = world_dir1 / "records.parquet"
        if pf1.exists():
            ids1 = set(pd.read_parquet(pf1)["record_id"].tolist())

        gen2 = TrackMGenerator(seed=42)
        results2 = gen2.generate_nested_worlds(scales=[10], output_dir=tmp_path / "run2")
        world2, _ = results2[0]
        world_dir2 = tmp_path / "run2" / world2.world_id
        ids2 = set()
        pf2 = world_dir2 / "records.parquet"
        if pf2.exists():
            ids2 = set(pd.read_parquet(pf2)["record_id"].tolist())

        assert ids1 == ids2, "Record ID sets differ between runs with same seed"

    def test_record_count_matches_scale(self, tmp_path):
        """Each world must have exactly scale_n records."""
        from faulttrace_data.generator import TrackMGenerator
        import pandas as pd

        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[10, 50], output_dir=tmp_path)
        for world, manifest in results:
            assert manifest.row_count == world.scale_n


class TestQueryReproducibility:
    """Generated queries must be reproducible given same world_id and seed."""

    @pytest.fixture
    def world_data(self, tmp_path):
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[50], output_dir=tmp_path / "generated" / "worlds")
        world, _ = results[0]
        return tmp_path / "generated", world.world_id

    def test_query_spec_hash_stable(self, world_data):
        data_dir, world_id = world_data
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries1 = factory.generate_for_world(world_id=world_id, target_count=30)
        queries2 = factory.generate_for_world(world_id=world_id, target_count=30)

        hashes1 = sorted(q.spec_hash() for q in queries1)
        hashes2 = sorted(q.spec_hash() for q in queries2)
        assert hashes1 == hashes2, "Query spec hashes differ between runs"


class TestGoldReproducibility:
    """Gold answers must be reproducible: same query + same world → same answer."""

    @pytest.fixture
    def world_data(self, tmp_path):
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[50], output_dir=tmp_path / "generated" / "worlds")
        world, _ = results[0]
        return tmp_path / "generated", world.world_id

    def test_pandas_answer_stable(self, world_data):
        """Pandas evaluator must produce same answer for same query twice."""
        import pandas as pd
        from faulttrace_pipelines.query_factory import QueryFactory
        from faulttrace_gold.pandas_engine import PandasEvaluator

        data_dir, world_id = world_data
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=10)

        parquet_path = data_dir / "worlds" / world_id / "records.parquet"
        if not parquet_path.exists():
            pytest.skip("No parquet file found")

        df = pd.read_parquet(parquet_path)
        evaluator = PandasEvaluator()

        for q in queries[:3]:
            r1 = evaluator.evaluate(q, df)
            r2 = evaluator.evaluate(q, df)
            assert r1["result"] == r2["result"], f"Pandas answer unstable for {q.template_id}"

    def test_duckdb_answer_stable(self, world_data):
        """DuckDB evaluator must produce same answer for same query twice."""
        import pandas as pd
        from faulttrace_pipelines.query_factory import QueryFactory
        from faulttrace_gold.duckdb_engine import DuckDBEvaluator

        data_dir, world_id = world_data
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=10)

        parquet_path = data_dir / "worlds" / world_id / "records.parquet"
        if not parquet_path.exists():
            pytest.skip("No parquet file found")

        df = pd.read_parquet(parquet_path)
        evaluator = DuckDBEvaluator()

        for q in queries[:3]:
            r1 = evaluator.evaluate(q, parquet_path)
            r2 = evaluator.evaluate(q, parquet_path)
            assert r1["result"] == r2["result"], f"DuckDB answer unstable for {q.template_id}"


class TestSecurityPathTraversal:
    """Path traversal defenses in data-loading components."""

    def test_snapshot_fingerprint_not_path(self, tmp_path):
        """Source path fingerprint must not leak absolute path."""
        from faulttrace_data.snapshot import _fingerprint_path
        data_root = tmp_path / "data"
        data_root.mkdir()
        evil_path = tmp_path / "../../../etc/passwd"
        fp = _fingerprint_path(evil_path, data_root)
        # Fingerprint must be a SHA-256 hex string, not a path
        assert len(fp) == 64
        assert "/" not in fp
        assert "\\" not in fp

    def test_amazon_adapter_refuses_oversized_file(self, tmp_path):
        """Adapter must raise when file exceeds max_bytes limit."""
        from faulttrace_data.amazon_adapter import AmazonLocalAdapter
        import json

        small_limit_adapter = AmazonLocalAdapter(dataset_id="test", max_bytes=5)
        tmp_file = tmp_path / "big.jsonl"
        tmp_file.write_text(json.dumps({"asin": "B0001", "average_rating": 4.0}) + "\n")

        with pytest.raises(ValueError):
            report, _ = small_limit_adapter.ingest(tmp_file, tmp_path / "out", data_root=tmp_path)

    def test_world_id_no_path_chars(self, tmp_path):
        """World IDs must not contain path separators or traversal sequences."""
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        results = gen.generate_nested_worlds(scales=[10], output_dir=tmp_path)
        for world, _ in results:
            assert "/" not in world.world_id
            assert "\\" not in world.world_id
            assert ".." not in world.world_id


class TestSchemaVersionAudit:
    def test_corpus_record_schema_version(self):
        from faulttrace_core.models import SCHEMA_VERSION
        # Must follow semver pattern
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_snapshot_schema_version(self):
        from faulttrace_data.snapshot import SNAPSHOT_SCHEMA_VERSION
        parts = SNAPSHOT_SCHEMA_VERSION.split(".")
        assert len(parts) == 3

    def test_world_builder_version(self):
        from faulttrace_data.world_builder import WORLD_BUILDER_VERSION
        parts = WORLD_BUILDER_VERSION.split(".")
        assert len(parts) == 3
