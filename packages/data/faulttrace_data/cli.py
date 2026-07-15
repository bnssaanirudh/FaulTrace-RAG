"""
CLI entry point for FaultTrace data commands — Prompt 2 extended.

Usage:
    faulttrace data seed --seed 42 --scales 10,50,200,1000
    faulttrace data ingest-amazon --input PATH --dataset-id ID --output DATA_ROOT
    faulttrace data inspect --dataset-id ID
    faulttrace data validate --dataset-id ID

    faulttrace worlds build --dataset-id DATASET --config configs/worlds/demo.yaml
    faulttrace worlds verify-nested --group demo-seed-42
    faulttrace worlds summarize --world-id WORLD_ID

    faulttrace gold validate --world-id WORLD_ID
    faulttrace query generate --world-id WORLD_ID
    faulttrace query pack --world-id WORLD_ID --count 300
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="faulttrace",
    help="FaultTrace-RAG CLI: data seeding, gold validation, and query generation.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

data_app = typer.Typer(help="Data generation and ingestion commands.")
gold_app = typer.Typer(help="Gold engine commands.")
query_app = typer.Typer(help="Query management commands.")
worlds_app = typer.Typer(help="World building and verification commands.")

app.add_typer(data_app, name="data")
app.add_typer(gold_app, name="gold")
app.add_typer(query_app, name="query")
app.add_typer(worlds_app, name="worlds")


# =============================================================================
# DATA COMMANDS
# =============================================================================


@data_app.command("seed")
def seed_data(
    seed: int = typer.Option(42, "--seed", "-s", help="RNG seed for determinism"),
    scales: str = typer.Option(
        "10,50,200,1000",
        "--scales",
        help="Comma-separated list of world scales to generate",
    ),
    output_dir: Path = typer.Option(
        Path("data/generated"),
        "--output-dir",
        help="Directory for generated data",
    ),
    fixtures: bool = typer.Option(True, "--fixtures/--no-fixtures", help="Also generate adversarial fixtures"),
    register_db: Optional[str] = typer.Option(
        None, "--register-db", help="SQLite DB path to register worlds (optional)"
    ),
) -> None:
    """Generate deterministic nested corpus worlds at specified scales."""
    from faulttrace_data.generator import TrackMGenerator

    scale_list = [int(s.strip()) for s in scales.split(",")]
    typer.echo(f"[faulttrace data seed] seed={seed}, scales={scale_list}")
    typer.echo(f"  Output directory: {output_dir}")

    generator = TrackMGenerator(seed=seed)

    # Generate nested worlds
    results = generator.generate_nested_worlds(
        scales=scale_list,
        output_dir=output_dir / "worlds",
    )

    typer.echo(f"\n  Generated {len(results)} worlds:")
    world_data = []
    for world, manifest in results:
        typer.echo(
            f"    [{world.world_id}] n={world.scale_n} "
            f"records={manifest.row_count} "
            f"parquet_hash={manifest.parquet_hash[:12]}..."
        )
        world_data.append(world.model_dump(mode="json"))

    # Generate adversarial fixtures
    if fixtures:
        fixture_dir = output_dir / "fixtures"
        counts = generator.generate_adversarial_fixtures(fixture_dir)
        typer.echo(f"\n  Adversarial fixtures: {counts}")

    # Register in DB if requested
    if register_db:
        _register_worlds_in_db(register_db, [w for w, _ in results])
        typer.echo(f"\n  Registered worlds in: {register_db}")

    # Write summary
    summary_path = output_dir / "seed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed,
        "scales": scale_list,
        "worlds": world_data,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    typer.echo(f"\n  Summary written to: {summary_path}")
    typer.echo("\n[faulttrace data seed] Done.")


@data_app.command("ingest-amazon")
def ingest_amazon(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to Amazon JSONL/CSV/Parquet file"),
    dataset_id: str = typer.Option(..., "--dataset-id", help="Logical dataset identifier"),
    output: Path = typer.Option(
        Path("data/snapshots"),
        "--output",
        help="Root directory for canonical Parquet output",
    ),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", help="Project data root for path fingerprinting (default: output parent)"
    ),
    license_note: str = typer.Option("", "--license-note", help="License/provenance note"),
    register_db: Optional[str] = typer.Option(None, "--register-db", help="SQLite DB to register snapshot"),
    max_bytes_mb: int = typer.Option(500, "--max-bytes-mb", help="Max uncompressed size limit (MB)"),
) -> None:
    """Ingest a local Amazon-style file into a canonical Parquet snapshot."""
    from faulttrace_data.amazon_adapter import AmazonLocalAdapter
    from faulttrace_data.snapshot import SnapshotRegistry

    if not input_path.exists():
        typer.echo(f"[ERROR] Input file not found: {input_path}", err=True)
        raise typer.Exit(1)

    # Security: refuse absolute paths pointing outside data_root
    if data_root is None:
        data_root = output.parent

    typer.echo(f"[faulttrace data ingest-amazon] dataset_id={dataset_id}")
    typer.echo(f"  Input: {input_path.name} (size={input_path.stat().st_size:,} bytes)")
    typer.echo(f"  Output: {output}")

    adapter = AmazonLocalAdapter(
        dataset_id=dataset_id,
        max_bytes=max_bytes_mb * 1024 * 1024,
    )

    producing_cmd = f"faulttrace data ingest-amazon --input {input_path.name} --dataset-id {dataset_id}"

    try:
        report, snapshot = adapter.ingest(
            source_path=input_path,
            output_root=output,
            data_root=data_root,
            license_note=license_note,
            producing_command=producing_cmd,
        )
    except ValueError as e:
        typer.echo(f"[ERROR] Ingestion failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n  Snapshot ID:  {snapshot.snapshot_id}")
    typer.echo(f"  Rows read:    {report.total_rows_read:,}")
    typer.echo(f"  Accepted:     {report.accepted_count:,}")
    typer.echo(f"  Rejected:     {report.rejected_count:,}")
    typer.echo(f"  Duplicates:   {report.duplicate_count:,}")
    typer.echo(f"  Malformed:    {report.malformed_count:,}")
    if report.rejection_reasons:
        typer.echo(f"  Rejection reasons: {report.rejection_reasons}")
    typer.echo(f"  Parquet output: {report.parquet_output_path}")

    # Register snapshot
    registry_path = data_root / "manifests" / "snapshots.jsonl"
    registry = SnapshotRegistry(registry_path)
    registry.register(snapshot)
    typer.echo(f"\n  Registered in: {registry_path}")

    typer.echo("\n[faulttrace data ingest-amazon] Done.")


@data_app.command("inspect")
def inspect_dataset(
    dataset_id: str = typer.Option(..., "--dataset-id", help="Dataset ID to inspect"),
    data_root: Path = typer.Option(Path("data"), "--data-root", help="Project data root"),
) -> None:
    """Inspect all snapshots for a dataset."""
    from faulttrace_data.snapshot import SnapshotRegistry

    registry_path = data_root / "manifests" / "snapshots.jsonl"
    registry = SnapshotRegistry(registry_path)
    snapshots = registry.list_snapshots(dataset_id=dataset_id)

    if not snapshots:
        typer.echo(f"No snapshots found for dataset '{dataset_id}'")
        return

    typer.echo(f"\n[faulttrace data inspect] Dataset: {dataset_id} ({len(snapshots)} snapshots)")
    for snap in snapshots:
        typer.echo(f"\n  Snapshot: {snap.snapshot_id}")
        typer.echo(f"    Source type:  {snap.source_type}")
        typer.echo(f"    Rows:         {snap.row_count:,}")
        typer.echo(f"    Accepted:     {snap.accepted_count:,}")
        typer.echo(f"    Rejected:     {snap.rejected_count:,}")
        typer.echo(f"    Duplicates:   {snap.duplicate_count:,}")
        typer.echo(f"    Malformed:    {snap.malformed_count:,}")
        typer.echo(f"    Created:      {snap.created_at}")
        typer.echo(f"    Active:       {snap.active}")
        if snap.missingness.price_null_ratio > 0:
            typer.echo(f"    Price null:   {snap.missingness.price_null_ratio:.1%}")


@data_app.command("validate")
def validate_dataset(
    dataset_id: str = typer.Option(..., "--dataset-id", help="Dataset ID to validate"),
    data_root: Path = typer.Option(Path("data"), "--data-root", help="Project data root"),
) -> None:
    """Validate all snapshots for a dataset (hash checks + artifact presence)."""
    from faulttrace_data.snapshot import SnapshotRegistry

    registry_path = data_root / "manifests" / "snapshots.jsonl"
    registry = SnapshotRegistry(registry_path)
    snapshots = registry.list_snapshots(dataset_id=dataset_id)

    if not snapshots:
        typer.echo(f"No snapshots found for dataset '{dataset_id}'")
        return

    all_valid = True
    for snap in snapshots:
        result = registry.validate(snap.snapshot_id, data_root)
        status = result["status"]
        icon = "✓" if status == "valid" else "✗"
        typer.echo(f"  [{icon}] {snap.snapshot_id[:12]}... — {status}")
        for check in result.get("checks", []):
            check_icon = "✓" if check["status"] == "pass" else "✗"
            typer.echo(f"      [{check_icon}] {check['check']}")
        if status != "valid":
            all_valid = False

    if not all_valid:
        raise typer.Exit(1)
    typer.echo(f"\n[faulttrace data validate] All {len(snapshots)} snapshots valid.")


# =============================================================================
# WORLDS COMMANDS
# =============================================================================


@worlds_app.command("build")
def build_worlds(
    dataset_id: str = typer.Option(..., "--dataset-id", help="Dataset ID to build worlds from"),
    config: Path = typer.Option(
        Path("configs/worlds/demo.yaml"),
        "--config",
        help="worlds.yaml configuration file",
    ),
    data_dir: Path = typer.Option(Path("data/generated"), "--data-dir"),
    output_dir: Path = typer.Option(Path("data/generated/worlds"), "--output-dir"),
) -> None:
    """Build nested corpus worlds from a dataset using a worlds.yaml config."""
    from faulttrace_data.world_builder import WorldBuilder, load_worlds_config

    if not config.exists():
        typer.echo(f"[ERROR] Config not found: {config}", err=True)
        raise typer.Exit(1)

    cfg = load_worlds_config(config)
    producing_cmd = f"faulttrace worlds build --dataset-id {dataset_id} --config {config}"

    # Find matching groups
    groups = [g for g in cfg.groups if g.dataset_id == dataset_id or dataset_id == "all"]
    if not groups:
        typer.echo(f"No groups found for dataset_id='{dataset_id}' in config", err=True)
        raise typer.Exit(1)

    for group in groups:
        typer.echo(f"\n[faulttrace worlds build] group={group.group_id} seed={group.seed}")

        # Find source Parquet
        source_parquet = data_dir / "worlds" / f"world_s{group.seed}_n{max(s.n for s in group.scales)}"
        if not source_parquet.exists():
            # Try the largest existing world
            world_dirs = sorted(data_dir.glob(f"world_s{group.seed}_n*"))
            if not world_dirs:
                typer.echo(f"  [WARN] No source data found for seed={group.seed}. Run 'data seed' first.")
                continue
            source_parquet = world_dirs[-1]

        builder = WorldBuilder(
            source_parquet=source_parquet / "records.parquet",
            dataset_id=group.dataset_id,
            seed=group.seed,
            group_id=group.group_id,
        )

        scales = [s.n for s in group.scales]
        manifests = builder.build_nested_worlds(
            scales=scales,
            output_dir=output_dir,
            policy=group.scales[0].policy,
            producing_command=producing_cmd,
        )

        for m in manifests:
            typer.echo(f"  [{m.world_id}] n={m.scale_n} hash={m.parquet_hash[:12]}...")

        # Verify nestedness
        verify_result = builder.verify_nestedness(manifests)
        typer.echo(f"\n  Nestedness: {verify_result['status']}")

        if group.include_adversarial:
            adv_slices = builder.build_adversarial_slices(output_dir)
            typer.echo(f"  Adversarial slices: {list(adv_slices.keys())}")

    typer.echo("\n[faulttrace worlds build] Done.")


@worlds_app.command("verify-nested")
def verify_nested(
    group: str = typer.Option(..., "--group", help="World group ID to verify"),
    output_dir: Path = typer.Option(Path("data/generated/worlds"), "--output-dir"),
) -> None:
    """Cryptographically verify nestedness for a world group."""
    import json
    from faulttrace_data.world_builder import WorldManifestV2

    # Find all manifests for this group
    manifests = []
    for world_dir in sorted(output_dir.glob("world_*")):
        manifest_path = world_dir / "manifest.json"
        if manifest_path.exists():
            try:
                m = WorldManifestV2(**json.loads(manifest_path.read_text()))
                if m.group_id == group:
                    manifests.append(m)
            except Exception:
                pass

    if len(manifests) < 2:
        typer.echo(f"Found {len(manifests)} worlds for group '{group}'. Need ≥2 for nestedness check.")
        return

    manifests.sort(key=lambda m: m.scale_n)

    # Use WorldBuilder for verification
    from faulttrace_data.world_builder import WorldBuilder
    builder = WorldBuilder(
        source_parquet=Path(manifests[0].parquet_path),
        dataset_id=manifests[0].dataset_id,
        seed=manifests[0].seed,
        group_id=group,
    )
    result = builder.verify_nestedness(manifests)

    typer.echo(f"\n[faulttrace worlds verify-nested] group={group}")
    typer.echo(f"  Status: {result['status']}")
    typer.echo(f"  Worlds checked: {result['worlds_checked']}")
    typer.echo(f"  Pairs verified: {result['pairs_verified']}")
    for check in result.get("checks", []):
        icon = "✓" if check.get("status") == "nested_verified" else "✗"
        typer.echo(f"  [{icon}] {check.get('pair', ['-', '-'])}: {check.get('status', 'unknown')}")

    if result["status"] != "nested":
        raise typer.Exit(1)


@worlds_app.command("summarize")
def summarize_world(
    world_id: str = typer.Option(..., "--world-id", help="World ID to summarize"),
    output_dir: Path = typer.Option(Path("data/generated/worlds"), "--output-dir"),
) -> None:
    """Show summary statistics for a world."""
    import json
    from faulttrace_data.world_builder import WorldBuilder, WorldManifestV2

    manifest_path = output_dir / world_id / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"[ERROR] Manifest not found: {manifest_path}", err=True)
        raise typer.Exit(1)

    manifest = WorldManifestV2(**json.loads(manifest_path.read_text()))
    builder = WorldBuilder(
        source_parquet=Path(manifest.parquet_path),
        dataset_id=manifest.dataset_id,
        seed=manifest.seed,
    )
    summary = builder.summarize_world(manifest)

    typer.echo(f"\n[faulttrace worlds summarize] {world_id}")
    typer.echo(json.dumps(summary, indent=2, default=str))


# =============================================================================
# GOLD COMMANDS
# =============================================================================


@gold_app.command("validate")
def validate_gold(
    world_id: str = typer.Option(..., "--world-id", help="World ID to validate"),
    data_dir: Path = typer.Option(Path("data/generated"), "--data-dir"),
    output_dir: Path = typer.Option(Path("artifacts/gold_reports"), "--output-dir"),
) -> None:
    """Validate all generated queries using dual Pandas + DuckDB gold engines."""
    from faulttrace_gold.validator import validate_world_queries

    typer.echo(f"[faulttrace gold validate] world_id={world_id}")
    report = validate_world_queries(world_id=world_id, data_dir=data_dir, output_dir=output_dir)

    typer.echo(f"  Total queries: {report['total']}")
    typer.echo(f"  Agreed: {report['agreed']}")
    typer.echo(f"  Disagreed: {report['disagreed']}")
    typer.echo(f"  Report: {report.get('report_path', 'N/A')}")

    if report["disagreed"] > 0:
        typer.echo("\n  [WARNING] Disagreements found! See report for details.")
        raise typer.Exit(1)

    typer.echo("\n[faulttrace gold validate] All queries agree.")


# =============================================================================
# QUERY COMMANDS
# =============================================================================


@query_app.command("generate")
def generate_queries(
    world_id: str = typer.Option(..., "--world-id", help="World ID to generate queries for"),
    data_dir: Path = typer.Option(Path("data/generated"), "--data-dir"),
    count: int = typer.Option(60, "--count", help="Target number of queries to generate"),
    output_dir: Path = typer.Option(Path("artifacts/queries"), "--output-dir"),
    register_db: Optional[str] = typer.Option(None, "--register-db"),
) -> None:
    """Generate procedural queries for a corpus world."""
    from faulttrace_pipelines.query_factory import QueryFactory

    typer.echo(f"[faulttrace query generate] world_id={world_id}, count={count}")

    factory = QueryFactory(data_dir=data_dir)
    queries = factory.generate_for_world(world_id=world_id, target_count=count)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"queries_{world_id}.jsonl"
    with open(out_path, "w") as f:
        for q in queries:
            f.write(json.dumps(q.model_dump(mode="json"), default=str) + "\n")

    typer.echo(f"  Generated {len(queries)} queries -> {out_path}")

    if register_db:
        _register_queries_in_db(register_db, queries)
        typer.echo(f"  Registered in: {register_db}")

    typer.echo("[faulttrace query generate] Done.")


@query_app.command("pack")
def build_query_pack(
    world_id: str = typer.Option(..., "--world-id", help="World ID for benchmark pack"),
    data_dir: Path = typer.Option(Path("data/generated"), "--data-dir"),
    count: int = typer.Option(300, "--count", help="Total queries across all families"),
    output_dir: Path = typer.Option(Path("artifacts/query_packs"), "--output-dir"),
    validate_gold_flag: bool = typer.Option(True, "--validate/--no-validate", help="Run dual gold validation"),
    release: bool = typer.Option(False, "--release", help="Mark pack as gold_ready (requires zero disagreements)"),
) -> None:
    """Build a research benchmark pack with balanced queries across families and splits."""
    from faulttrace_pipelines.query_factory import QueryFactory, BenchmarkPack

    typer.echo(f"[faulttrace query pack] world_id={world_id}, count={count}")

    factory = QueryFactory(data_dir=data_dir)
    pack = factory.build_benchmark_pack(
        world_id=world_id,
        total_count=count,
        validate_gold=validate_gold_flag,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / f"pack_{world_id}.json"
    pack_path.write_text(json.dumps(pack.model_dump(mode="json"), indent=2, default=str))

    typer.echo(f"\n  Pack ID: {pack.pack_id}")
    typer.echo(f"  Total queries: {pack.total_count}")
    typer.echo(f"  Gold agreed: {pack.agreed_count}")
    typer.echo(f"  Gold disagreed: {pack.disagreed_count}")
    typer.echo(f"  Split counts: dev={pack.dev_count} val={pack.val_count} test={pack.test_count}")
    typer.echo(f"  Gold ready: {pack.gold_ready}")

    if release and not pack.gold_ready:
        typer.echo("[ERROR] Cannot release: pack has disagreements. Fix gold engine first.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n  Pack written to: {pack_path}")
    typer.echo("[faulttrace query pack] Done.")


# =============================================================================
# Shared helpers
# =============================================================================


def _register_worlds_in_db(db_path: str, worlds: list) -> None:
    """Register worlds in the SQLite database."""
    try:
        from faulttrace_api.database import get_engine
        from faulttrace_api.repositories.worlds import WorldRepository
        engine = get_engine(db_path)
        repo = WorldRepository(engine)
        for world in worlds:
            repo.upsert(world)
    except ImportError:
        pass


def _register_queries_in_db(db_path: str, queries: list) -> None:
    """Register queries in the SQLite database."""
    try:
        from faulttrace_api.database import get_engine
        from faulttrace_api.repositories.queries import QueryRepository
        engine = get_engine(db_path)
        repo = QueryRepository(engine)
        for q in queries:
            repo.upsert(q)
    except ImportError:
        pass


if __name__ == "__main__":
    app()
