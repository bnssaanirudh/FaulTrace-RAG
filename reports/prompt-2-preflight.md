# FaultTrace-RAG Prompt 2 — Pre-Change Baseline

**Date**: 2026-07-15  
**Recorded by**: Lead data systems engineer  
**Purpose**: Establish a verified pre-change state before any Prompt 2 modifications.

---

## Test Baseline

```
Command: python -m pytest apps/api/tests/ -v --tb=short
Date:    2026-07-15
Result:  91 passed, 1 skipped in 4.03s
```

| Suite | Tests | Result |
|-------|-------|--------|
| test_core_contracts.py | 29 | ✅ Pass |
| test_generator.py | 11 | ✅ Pass |
| test_gold_engine.py | 20 | ✅ Pass |
| test_pipelines.py | 30 | ✅ Pass (1 skipped) |
| test_smoke_e2e.py | 8 | ✅ Pass |

Skipped: `test_attribution_shapley_sum_near_1` — marked skip in source.

---

## Schema Audit

### SCHEMA_VERSION
- Current: `"1.0.0"` in `faulttrace_core/models.py`
- GENERATOR_VERSION: `"1.0.0"` in `faulttrace_data/generator.py`
- No migration needed; all models are Pydantic v2 with optional backward-compatible additions.

### CorpusRecord fields present
| Field | Present | Hash-stable |
|-------|---------|-------------|
| record_id | ✅ | ✅ |
| source | ✅ | — |
| source_record_id | ✅ | ✅ |
| world_id | ✅ | ✅ |
| product_id | ✅ | ✅ |
| category | ✅ | ✅ |
| title | ✅ | ✅ |
| brand | ✅ | ✅ |
| rating | ✅ | ✅ |
| event_time | ✅ | ✅ |
| verified_purchase | ✅ | ✅ |
| price | ✅ | — (optional) |
| raw_payload_hash | ✅ | ✅ |
| schema_version | ✅ | — |

### WorldManifest fields
| Field | Present | Gap vs Prompt 2 Requirement |
|-------|---------|----------------------------|
| world_id | ✅ | — |
| generator_version | ✅ | — |
| seed | ✅ | — |
| scale_n | ✅ | — |
| row_count | ✅ | — |
| parquet_hash | ✅ | — |
| jsonl_hash | ✅ | — |
| summary_stats | ✅ | — |
| parent_world_id | ✅ | — |
| created_at | ✅ | — |
| producing_command | ❌ | Add (optional, default=None) |
| config_hash | ❌ | Add (optional, default=None) |
| parent_artifact_refs | ❌ | Add (optional, default=[]) |
| logical_id | ❌ | world_id serves this role |

### QuerySpec fields
| Field | Present | Gap |
|-------|---------|-----|
| query_id | ✅ | — |
| family | ✅ | — |
| natural_language_question | ✅ | — |
| scope_predicate | ✅ | — |
| fact_spec | ✅ | — |
| aggregation_spec | ✅ | — |
| tolerance | ✅ | — |
| world_id | ✅ | — |
| template_id | ✅ | — |
| spec_hash() | ✅ | — |
| dataset_snapshot_id | ❌ | Add (optional) |
| difficulty | ❌ | Add (optional) |
| split | ❌ | Add for dev/val/test |

---

## Query Factory Audit

| Family | Current Templates | Target (Prompt 2) |
|--------|------------------|-------------------|
| COUNT | 10 | 20 |
| MEAN/SUM | 8 | 15 |
| PROPORTION | 6 | 20 |
| COMPARISON | 6 | 15 |
| TOP-K | 6 | 15 |
| TREND | 6 | 15 |
| **Total** | **42** | **100** |

Current generated count per world: ~60 queries.
Target research demo: ≥300 validated queries across scales.

---

## Artifact Hashes (at baseline)

These are the hashes of key source files before any modification:

| File | SHA-256 prefix |
|------|----------------|
| packages/core/faulttrace_core/models.py | See content_hash below |
| packages/data/faulttrace_data/generator.py | See content_hash below |
| packages/pipelines/faulttrace_pipelines/query_factory.py | See content_hash below |

(Full hashes recorded in reports/prompt-2-completion.md after changes.)

---

## Known Limitations at Baseline (from KNOWN_LIMITATIONS.md)

- KL-001: Synthetic corpus only (no real Amazon data) — Prompt 2 adds adapter
- KL-002: No LLM — Prompt 2 scope does not add LLM
- KL-003: No oracle lattice — Prompt 3 scope
- KL-004: No paper experiments — Prompt 4-5 scope
- KL-005: Node.js 20+ required for frontend — unchanged
- KL-006: Python 3.13 in use — compatible
- KL-007: SQLite single-writer — unchanged
- KL-008: No auth — unchanged

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking existing tests during model changes | High | Only add optional fields; existing defaults preserved |
| Query count <300 for demo | Medium | Generate across all 4 scales, aggregate packs |
| Pandas/DuckDB parity for new DSL predicates | Medium | Property-based tests for each predicate type |
| Path traversal in Amazon adapter | High | Whitelist-only path resolution; no user-controlled paths |
| Large fixture decompression bombs | Medium | Configurable size limit (default 50MB compressed) |

---

## Conclusion

Baseline is clean and stable. Prompt 2 modifications will:
1. Add backward-compatible optional fields only to existing models
2. Add entirely new modules (no modifications to working gold engines until WP7)
3. Add new test files without removing existing tests
4. All 91 existing tests must pass after every work package
