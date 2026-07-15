"""
Tests for the expanded query factory (WP6).

Tests cover:
- Template count: exactly 100 templates registered
- Family distribution: 6 families present
- All templates generate valid QuerySpec objects
- BenchmarkPack: total_count > 0, splits sum correctly
- Duplicate detection: spec hashes are unique
- Dev/val/test split is deterministic
- Template registry: all entries have required dimensions
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from faulttrace_core.models import QueryFamily


class TestTemplateRegistry:
    def test_total_template_count(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY, ALL_TEMPLATES
        total = sum(len(v) for v in ALL_TEMPLATES.values())
        assert total == 100, f"Expected 100 templates, got {total}"

    def test_all_six_families_present(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        families = {e.family for e in TEMPLATE_REGISTRY._entries.values()}
        expected = {"count", "mean", "proportion", "comparison", "top_k", "trend"}
        assert expected.issubset(families)

    def test_count_family_has_20_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.COUNT]) == 20

    def test_mean_family_has_15_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.MEAN]) == 15

    def test_proportion_family_has_20_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.PROPORTION]) == 20

    def test_comparison_family_has_15_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.COMPARISON]) == 15

    def test_topk_family_has_15_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.TOP_K]) == 15

    def test_trend_family_has_15_templates(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        assert len(ALL_TEMPLATES[QueryFamily.TREND]) == 15

    def test_all_template_ids_unique(self):
        from faulttrace_pipelines.query_factory import ALL_TEMPLATES
        ids = []
        for templates in ALL_TEMPLATES.values():
            for entry in templates:
                ids.append(entry[0])
        assert len(ids) == len(set(ids)), "Duplicate template IDs found"

    def test_every_entry_has_dimensions(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        required_dims = {"difficulty", "selectivity", "null_risk", "tie_risk", "temporal_risk"}
        for entry in TEMPLATE_REGISTRY._entries.values():
            dims = set(entry.dimensions.keys())
            assert required_dims.issubset(dims), f"Template {entry.template_id} missing dimensions"

    def test_registry_summary(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        summary = TEMPLATE_REGISTRY.summary()
        assert summary["total_templates"] == 100
        assert "by_family" in summary
        assert "by_difficulty" in summary

    def test_list_by_family(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        count_entries = TEMPLATE_REGISTRY.list_by_family(QueryFamily.COUNT)
        assert len(count_entries) == 20

    def test_list_by_difficulty(self):
        from faulttrace_pipelines.query_factory import TEMPLATE_REGISTRY
        easy = TEMPLATE_REGISTRY.list_by_difficulty("easy")
        medium = TEMPLATE_REGISTRY.list_by_difficulty("medium")
        adversarial = TEMPLATE_REGISTRY.list_by_difficulty("adversarial")
        assert len(easy) > 0
        assert len(medium) > 0
        assert len(adversarial) > 0


class TestQueryGeneration:
    """Test query generation against a real generated world."""

    @pytest.fixture
    def world_dir(self, tmp_path):
        """Create a minimal synthetic world for testing."""
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        worlds_dir = tmp_path / "generated" / "worlds"
        results = gen.generate_nested_worlds(scales=[200], output_dir=worlds_dir)
        world, manifest = results[0]
        return tmp_path / "generated", world.world_id

    def test_generate_for_world_returns_queries(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=60)
        assert len(queries) > 0

    def test_queries_cover_all_families(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=120)
        families = {q.family.value for q in queries}
        expected = {"count", "mean", "proportion", "comparison", "top_k", "trend"}
        assert expected.issubset(families), f"Missing families: {expected - families}"

    def test_spec_hashes_unique(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=60)
        hashes = [q.spec_hash() for q in queries]
        assert len(hashes) == len(set(hashes)), "Duplicate spec hashes detected"

    def test_all_queries_have_valid_world_id(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=60)
        for q in queries:
            assert q.world_id == world_id

    def test_queries_have_template_id(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=60)
        for q in queries:
            assert q.template_id != "manual"
            assert q.template_id  # non-empty

    def test_query_natural_language_nonempty(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        queries = factory.generate_for_world(world_id=world_id, target_count=60)
        for q in queries:
            assert len(q.natural_language_question) >= 10


class TestBenchmarkPack:
    @pytest.fixture
    def world_dir(self, tmp_path):
        from faulttrace_data.generator import TrackMGenerator
        gen = TrackMGenerator(seed=42)
        worlds_dir = tmp_path / "generated" / "worlds"
        results = gen.generate_nested_worlds(scales=[200], output_dir=worlds_dir)
        world, _ = results[0]
        return tmp_path / "generated", world.world_id

    def test_pack_created(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        assert pack.total_count > 0
        assert pack.world_id == world_id

    def test_splits_sum_to_total(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        assert pack.dev_count + pack.val_count + pack.test_count == pack.total_count

    def test_split_proportions(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack = factory.build_benchmark_pack(world_id=world_id, total_count=100, validate_gold=False)
        # Should be approximately 80/10/10 — allow tolerance
        assert pack.dev_count >= pack.total_count * 0.6  # at least 60%
        assert pack.test_count >= 0

    def test_split_deterministic(self, world_dir):
        """Two packs with same world_id should have same splits (deterministic via spec_hash)."""
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack1 = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        pack2 = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        assert sorted(pack1.dev_query_ids) == sorted(pack2.dev_query_ids)

    def test_pack_has_family_distribution(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        assert len(pack.count_by_family) > 0

    def test_no_duplicate_spec_hashes(self, world_dir):
        data_dir, world_id = world_dir
        from faulttrace_pipelines.query_factory import QueryFactory
        factory = QueryFactory(data_dir=data_dir)
        pack = factory.build_benchmark_pack(world_id=world_id, total_count=60, validate_gold=False)
        # All IDs should be unique (duplicates removed)
        all_ids = pack.dev_query_ids + pack.val_query_ids + pack.test_query_ids
        assert len(all_ids) == len(set(all_ids))
