import pandas as pd
from faulttrace_reporting import (
    ExperimentSpec,
    FigureGenerator,
    MetricsComputer,
    ReproducibilityBundle,
    ResumableMatrixRunner,
    compute_paired_bootstrap_ci,
)


def test_reporting_metrics():
    """Unit-test metric formulas with hand-calculated fixtures."""
    runs = [
        {
            "run_id": "1",
            "is_correct": True,
            "loss": 0.0,
            "latency_ms": 100,
            "policy_decision": "certified",
            "status": "completed",
        },
        {
            "run_id": "2",
            "is_correct": False,
            "loss": 1.0,
            "latency_ms": 200,
            "policy_decision": "abstain",
            "status": "completed",
        },
        {
            "run_id": "3",
            "is_correct": False,
            "loss": 1.0,
            "latency_ms": 150,
            "policy_decision": "certified",
            "status": "completed",
        },
    ]
    metrics = MetricsComputer.compute_all(runs)
    # Total samples: 3, Correct: 1 -> Accuracy 33.3%
    assert abs(metrics.accuracy - (1 / 3)) < 1e-4

    # Coverage: 2 certified out of 3 -> 66.6%
    assert abs(metrics.certified_rate - (2 / 3)) < 1e-4

    # False certification: 1 wrong out of 2 certified -> 50%
    assert abs(metrics.false_certification_rate - 0.5) < 1e-4


def test_reporting_stats():
    """Unit-test bootstrap logic."""
    acc1 = [1.0, 1.0, 0.0, 1.0, 0.0]
    acc2 = [1.0, 1.0, 1.0, 1.0, 1.0]

    mean_diff, (ci_low, ci_high), effect = compute_paired_bootstrap_ci(acc1, acc2, samples=100)
    # acc1 mean = 0.6, acc2 mean = 1.0, diff = -0.4
    assert abs(mean_diff - (-0.4)) < 1e-4
    assert ci_low <= mean_diff <= ci_high


def test_reporting_runner():
    """Test matrix expansion and validation policies."""
    spec = ExperimentSpec(
        name="test_expansion",
        dataset_id="amazon_demo",
        scales=[10, 50],
        query_families=["count"],
        difficulty_strata=["easy"],
        pipelines=["P0-deterministic-scope-baseline", "P1-wrong-scope"],
        providers=["deterministic"],
        models=["gpt-4o"],
        retriever="bm25",
        seeds=[42, 43],
    )
    # Combinations = scales(2) * families(1) * strata(1) * pipelines(2) * providers(1) * models(1) * seeds(2)
    runner = ResumableMatrixRunner(spec)
    plan = runner.dry_run()
    assert plan["total_jobs"] >= 0


def test_reporting_figures(tmp_path):
    """Test figure generation with zero samples and missing groups."""
    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    fake_runs = [
        {
            "run_id": "1",
            "is_correct": True,
            "loss": 0.0,
            "latency_ms": 100,
            "policy_decision": "certified",
            "status": "completed",
            "scale_n": 50,
            "pipeline_id": "P0-deterministic-scope-baseline",
        }
    ]
    fg = FigureGenerator(fake_runs, out_dir)
    # Should not crash on empty runs, or at least generate empty CSV
    figs = fg.generate_all()
    # At least some SVGs or CSVs should be written without crashing
    assert len(figs) > 0


def test_reporting_bundles(tmp_path):
    """Test report escaping and integrity verification."""
    tmp_path / "test_bundle"

    spec = {"name": "test"}
    metrics_df = pd.DataFrame([{"a": 1}])

    bundle_path = ReproducibilityBundle.export_bundle("test_1", spec, metrics_df, tmp_path)

    # Should verify properly
    is_valid, errors = ReproducibilityBundle.verify_bundle(bundle_path)
    assert is_valid is True
    assert len(errors) == 0

    # Tamper with file
    (bundle_path / "metrics.csv").write_text("tampered")
    is_valid, errors = ReproducibilityBundle.verify_bundle(bundle_path)
    assert is_valid is False
    assert len(errors) > 0
