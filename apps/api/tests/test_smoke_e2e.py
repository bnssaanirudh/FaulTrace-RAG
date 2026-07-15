"""
Smoke test: full end-to-end path through data generation, gold evaluation, and P0 pipeline.

This test exercises the complete vertical slice:
1. Generate N=50 corpus world
2. Generate 20 queries using QueryFactory
3. Validate queries against gold (both engines must agree)
4. Execute P0 pipeline on 5 queries with gold comparison
5. Verify artifact files were created
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from faulttrace_data.generator import TrackMGenerator
from faulttrace_pipelines.query_factory import QueryFactory
from faulttrace_gold.validator import GoldValidator
from faulttrace_pipelines.p0_baseline import P0DeterministicBaseline
from faulttrace_core.models import GoldAnswer, QuerySpec


@pytest.fixture(scope="module")
def world_data(tmp_path_factory):
    """Generate a world and return (world, manifest, df, parquet_path)."""
    tmp_dir = tmp_path_factory.mktemp("worlds")
    gen = TrackMGenerator(seed=42)
    world, manifest = gen.generate_world(n=50, output_dir=tmp_dir / "w1")
    
    parquet_path = Path(manifest.parquet_path)
    df = pd.read_parquet(parquet_path)
    return world, manifest, df, parquet_path


@pytest.fixture(scope="module")
def queries_with_gold(world_data, tmp_path_factory):
    """Generate queries and compute gold answers."""
    world, manifest, df, parquet_path = world_data
    tmp_dir = tmp_path_factory.mktemp("data")
    
    factory = QueryFactory(data_dir=tmp_dir)
    
    # Override data_dir to point to generated world
    worlds_dir = tmp_dir / "generated" / "worlds" / world.world_id
    worlds_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(parquet_path, worlds_dir / "records.parquet")
    
    factory = QueryFactory(data_dir=tmp_dir / "generated")
    queries = factory.generate_for_world(world_id=world.world_id, target_count=20, seed=42)
    
    validator = GoldValidator()
    results = []
    for q in queries:
        gold_result = validator.validate(q, df, worlds_dir / "records.parquet")
        results.append((q, gold_result))
    
    return results


def test_world_generation(world_data):
    """Generated world must have correct structure."""
    world, manifest, df, parquet_path = world_data
    
    assert world.scale_n == 50
    assert world.seed == 42
    assert manifest.row_count == 50
    assert len(df) == 50
    assert parquet_path.exists()
    
    # Check required columns
    required = {"record_id", "category", "brand", "rating", "price", 
                "verified_purchase", "helpful_votes", "event_time", "world_id"}
    assert required.issubset(set(df.columns))
    
    # Check world_id populated
    assert (df["world_id"] == world.world_id).all()


def test_query_generation(queries_with_gold):
    """At least 15 queries should be generated from 20 target."""
    assert len(queries_with_gold) >= 15, f"Only {len(queries_with_gold)} queries generated"


def test_gold_agreement_rate(queries_with_gold):
    """At least 90% of queries must have Pandas/DuckDB agreement."""
    agreed = sum(1 for _, gr in queries_with_gold if gr.agreed)
    total = len(queries_with_gold)
    rate = agreed / total if total > 0 else 0
    
    assert rate >= 0.90, (
        f"Gold agreement rate {rate:.0%} < 90%. "
        f"Disagreements: {[gr.diagnostic for _, gr in queries_with_gold if not gr.agreed]}"
    )


def test_all_query_families_covered(queries_with_gold):
    """All 6 families should be represented."""
    from faulttrace_core.models import QueryFamily
    
    families_found = {q.family for q, _ in queries_with_gold}
    
    expected = {
        QueryFamily.COUNT, QueryFamily.MEAN, QueryFamily.PROPORTION,
        QueryFamily.COMPARISON, QueryFamily.TOP_K, QueryFamily.TREND,
    }
    missing = expected - families_found
    assert not missing, f"Missing families: {missing}"


def test_p0_pipeline_execution(world_data, queries_with_gold, tmp_path):
    """P0 pipeline must execute without error on 5 queries."""
    _, _, df, parquet_path = world_data
    
    # Take up to 5 COUNT queries (simple family) for speed
    count_queries = [
        (q, gr) for q, gr in queries_with_gold
        if q.family.value == "count" and gr.gold_answer
    ][:5]
    
    if not count_queries:
        # Fall back to any agreed queries
        count_queries = [(q, gr) for q, gr in queries_with_gold if gr.gold_answer][:5]
    
    assert count_queries, "No queries with gold available for P0 test"
    
    pipeline = P0DeterministicBaseline(artifacts_dir=tmp_path / "runs")
    
    for q, gold_result in count_queries:
        gold = gold_result.gold_answer
        run, events, components = pipeline.run(
            query=q,
            df=df,
            gold_answer=gold,
            parquet_path=parquet_path,
        )
        
        assert run.status.value in ("completed", "failed"), f"Unexpected status: {run.status}"
        assert len(events) >= 5, f"Expected ≥5 trace events, got {len(events)}"
        assert len(components) >= 3, f"Expected ≥3 component outputs, got {len(components)}"
        
        if run.status.value == "completed":
            # Check artifacts were created
            run_dir = tmp_path / "runs" / run.run_id
            assert run_dir.exists(), f"Run dir not created: {run_dir}"
            assert (run_dir / "trace.jsonl").exists()
            assert (run_dir / "scope_output.parquet").exists()
            assert (run_dir / "extraction.parquet").exists()
            assert (run_dir / "aggregation_result.json").exists()


def test_p0_pipeline_correct_on_count(world_data, queries_with_gold, tmp_path):
    """P0 should get COUNT queries exactly right (it uses the gold engine)."""
    _, _, df, parquet_path = world_data
    
    count_queries = [
        (q, gr) for q, gr in queries_with_gold
        if q.family.value == "count" and gr.gold_answer is not None and gr.agreed
    ][:3]
    
    if not count_queries:
        pytest.skip("No count queries with gold available")
    
    pipeline = P0DeterministicBaseline(artifacts_dir=tmp_path / "runs2")
    
    correct = 0
    for q, gold_result in count_queries:
        run, _, _ = pipeline.run(
            query=q, df=df,
            gold_answer=gold_result.gold_answer,
            parquet_path=parquet_path,
        )
        if run.is_correct:
            correct += 1
    
    # P0 is deterministic so it should match gold always
    assert correct == len(count_queries), (
        f"P0 only got {correct}/{len(count_queries)} count queries correct"
    )


def test_trace_events_structure(world_data, queries_with_gold, tmp_path):
    """Trace events should have required fields and ordered stages."""
    _, _, df, _ = world_data
    
    agreed_queries = [(q, gr) for q, gr in queries_with_gold if gr.agreed and gr.gold_answer]
    if not agreed_queries:
        pytest.skip("No agreed queries available")
    
    q, gold_result = agreed_queries[0]
    pipeline = P0DeterministicBaseline(artifacts_dir=tmp_path / "runs3")
    run, events, _ = pipeline.run(query=q, df=df, gold_answer=gold_result.gold_answer)
    
    expected_stages = ["query_load", "scope_enumerate", "fact_extract", "aggregate", "validate", "persist"]
    event_stages = [ev.stage for ev in events]
    
    for expected_stage in expected_stages:
        assert expected_stage in event_stages, f"Missing stage: {expected_stage}"
    
    # Each event should have non-empty event_id
    for ev in events:
        assert ev.event_id, "Event must have an event_id"
        assert ev.run_id == run.run_id


def test_adversarial_empty_scope(world_data, tmp_path):
    """Empty scope should produce count=0 without error."""
    from faulttrace_core.models import (
        CountSpec, EqPredicate, FactSpec, QueryFamily, QuerySpec
    )
    _, _, df, parquet_path = world_data
    world, _, _, _ = world_data
    
    q = QuerySpec(
        family=QueryFamily.COUNT,
        natural_language_question="How many reviews have a rating of exactly 6 stars?",
        scope_predicate=EqPredicate(field="rating", value=6.0),
        fact_spec=FactSpec(fields=["record_id", "rating"]),
        aggregation_spec=CountSpec(),
        world_id=world.world_id,
        template_id="test_empty_scope",
    )
    
    validator = GoldValidator()
    gold_result = validator.validate(q, df, parquet_path)
    assert gold_result.agreed
    assert gold_result.gold_answer.answer_value == 0
    
    pipeline = P0DeterministicBaseline(artifacts_dir=tmp_path / "runs4")
    run, events, _ = pipeline.run(query=q, df=df, gold_answer=gold_result.gold_answer)
    assert run.status.value == "completed"
    assert int(run.answer) == 0
