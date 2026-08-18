"""
Locked Ablation Config Generators.
Generates deterministic configuration JSONs for standard benchmarks.
"""

from pathlib import Path

from faulttrace_reporting.experiments import ExperimentSpec


def generate_locked_ablations(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = []

    # 1. Demo Matrix (Small deterministic matrix)
    demo_spec = ExperimentSpec(
        name="demo_matrix",
        dataset_id="amazon_demo",
        scales=[10, 50],
        query_families=["count", "mean", "proportion", "comparison", "top_k", "trend"],
        difficulty_strata=["easy", "medium", "adversarial"],
        pipelines=[
            "P0-deterministic-scope-baseline",
            "P1-wrong-scope",
            "P2-wrong-facts",
            "P3-wrong-aggregation",
            "P4-compound-scope-facts",
            "P5-full-compound",
        ],
        providers=["deterministic"],
        models=["gpt-4o-mini"],
        retriever="bm25",
        top_k=10,
        chunk_size=500,
        context_budget=4000,
        batch_size=10,
        repair_policy="strict_exact_v1",
        certificate_policy="strict_exact_v1",
        seeds=[42],
        timeout_seconds=30.0,
        retries=3,
        cache_policy="use_cache",
        tags=["demo", "engineering_validation"],
    )
    configs.append(("demo_matrix.json", demo_spec))

    # 2. P4 vs P5 Repair Benefit (Ablation)
    p4_p5_spec = ExperimentSpec(
        name="ablation_p4_vs_p5",
        dataset_id="amazon_demo",
        scales=[50, 200, 1000],
        query_families=["count", "mean", "top_k"],
        difficulty_strata=["easy", "medium"],
        pipelines=["P4-compound-scope-facts", "P5-full-compound"],
        providers=["deterministic"],
        models=["gpt-4o-mini"],
        retriever="bm25",
        top_k=10,
        chunk_size=500,
        context_budget=4000,
        batch_size=10,
        repair_policy="strict_exact_v1",
        certificate_policy="strict_exact_v1",
        seeds=[42, 43, 44],
        timeout_seconds=30.0,
        retries=3,
        cache_policy="use_cache",
        tags=["ablation", "repair"],
    )
    configs.append(("ablation_p4_vs_p5.json", p4_p5_spec))

    # 3. Top-k Sweep (Ablation) - note: the config schema only supports single top_k.
    # To sweep, we must generate multiple configs.
    for k in [5, 10, 20, 50]:
        tk_spec = ExperimentSpec(
            name=f"ablation_topk_{k}",
            dataset_id="amazon_demo",
            scales=[200],
            query_families=["top_k", "count"],
            difficulty_strata=["medium"],
            pipelines=["P4-compound-scope-facts"],
            providers=["deterministic"],
            models=["gpt-4o-mini"],
            retriever="bm25",
            top_k=k,
            chunk_size=500,
            context_budget=4000,
            batch_size=10,
            repair_policy="strict_exact_v1",
            certificate_policy="strict_exact_v1",
            seeds=[42],
            timeout_seconds=30.0,
            retries=3,
            cache_policy="use_cache",
            tags=["ablation", "top_k"],
        )
        configs.append((f"ablation_topk_{k}.json", tk_spec))

    # Write configs
    generated = []
    for filename, spec in configs:
        out_path = output_dir / filename
        out_path.write_text(spec.model_dump_json(indent=2))
        generated.append(out_path)

    return generated


if __name__ == "__main__":
    out_dir = Path("configs/experiments")
    paths = generate_locked_ablations(out_dir)
    print(f"Generated {len(paths)} ablation configs in {out_dir}")
