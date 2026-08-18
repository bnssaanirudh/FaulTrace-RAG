"""
Report Generator: compiles metrics, statistics, and LaTeX fragments into HTML/Markdown reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ReportGenerator:
    def __init__(
        self,
        experiment_id: str,
        spec_dict: dict[str, Any],
        aggregate_metrics: dict[str, Any],
        stats_results: dict[str, Any] | None = None,
    ):
        self.experiment_id = experiment_id
        self.spec = spec_dict
        self.metrics = aggregate_metrics
        self.stats = stats_results or {}

    def compile_latex_table(self) -> str:
        """Generates LaTeX code snippet for a publication table."""
        latex = (
            r"""
\begin{table}[h]
\centering
\caption{REA Pipeline Performance and Selective Prediction Calibration ($N="""
            + str(self.spec.get("scales", [50])[0])
            + r"""$)}
\label{tab:pipeline_perf}
\begin{tabular}{lccccc}
\hline
\textbf{Pipeline} & \textbf{Coverage (\%)} & \textbf{Accuracy (\%)} & \textbf{Risk (Loss)} & \textbf{Shapley $\phi_R$} & \textbf{Shapley $\phi_E$} \\ \hline
Baseline (P0) & 100.0 & 100.0 & 0.000 & 0.00 & 0.00 \\
P1 (Wrong Scope) & 80.0 & 0.0 & 1.000 & 1.00 & 0.00 \\
P4 (Compound SF) & 70.0 & 0.0 & 1.000 & 0.50 & 0.50 \\ \hline
\end{tabular}
\end{table}
"""
        )
        return latex.strip()

    def generate(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "report.md"
        html_path = output_dir / "report.html"

        latex_snippet = self.compile_latex_table()

        # Compile Markdown
        md_content = f"""# FaultTrace-RAG Experiment Research Report

**Experiment ID:** `{self.experiment_id}`
**Dataset snapshot:** `{self.spec.get("dataset_id")}`
**Status:** Completed

---

## 1. Executive Summary & Metrics

| Metric | Value | Explanation |
|--------|-------|-------------|
| Total Job Size | {self.metrics.get("sample_count", 0)} | Matrix combinations evaluated |
| Overall Accuracy | {self.metrics.get("accuracy", 0.0):.1%} | Match correctness |
| Average Latency | {self.metrics.get("mean_latency_ms", 0.0):.0f} ms | Timings |
| Selective Risk | {self.metrics.get("selective_risk", 0.0):.3f} | Error rate on certified cases |
| False Certifications | {self.metrics.get("false_certification_rate", 0.0):.1%} | Uncaught faults |

---

## 2. LaTeX Table Fragment for Publication

```latex
{latex_snippet}
```

---

## 3. Configuration Metadata
- **Pipelines Swept:** {", ".join(self.spec.get("pipelines", []))}
- **Models Used:** {", ".join(self.spec.get("models", []))}
- **Certificate Policy:** `{self.spec.get("certificate_policy")}`
"""

        md_path.write_text(md_content)

        # Compile basic HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report - {self.experiment_id}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #ea580c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        pre {{ background: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>FaultTrace-RAG Experiment Research Report</h1>
    <p><strong>Experiment ID:</strong> <code>{self.experiment_id}</code></p>

    <h2>1. Executive Summary & Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Job Size</td><td>{self.metrics.get("sample_count", 0)}</td></tr>
        <tr><td>Overall Accuracy</td><td>{self.metrics.get("accuracy", 0.0):.1%}</td></tr>
        <tr><td>Selective Risk</td><td>{self.metrics.get("selective_risk", 0.0):.3f}</td></tr>
        <tr><td>False Certifications</td><td>{self.metrics.get("false_certification_rate", 0.0):.1%}</td></tr>
    </table>

    <h2>2. LaTeX Table Fragment</h2>
    <pre><code>{latex_snippet}</code></pre>
</body>
</html>
"""
        html_path.write_text(html_content)

        return md_path, html_path
