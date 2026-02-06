"""Report generation for accuracy assessment and model comparison.

Generates HTML reports with embedded plots: confusion matrix heatmaps,
per-class accuracy bar charts, error maps, and model comparison tables.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("lccomparison.report_generator")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    plt = None
    _MPL_AVAILABLE = False

from src.config_schema import CLASS_SCHEMA, CLASS_COLORS


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _fig_to_file(fig, path: Path) -> str:
    """Save a matplotlib figure to a file and return the filename."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    labels: list[int],
    title: str = "Confusion Matrix",
    output_path: Path | None = None,
) -> str | None:
    """Create a confusion matrix heatmap.

    Args:
        confusion_matrix: NxN matrix.
        labels: Class indices for rows/columns.
        title: Plot title.
        output_path: If provided, save to file; otherwise return base64.

    Returns:
        Base64 PNG string or filename.
    """
    if not _MPL_AVAILABLE:
        return None

    cm = np.array(confusion_matrix)
    idx_to_name = {v: k for k, v in CLASS_SCHEMA.items()}
    names = [idx_to_name.get(l, str(l)) for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))

    # Normalize for color intensity
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells with counts
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Row-normalized proportion")

    if output_path:
        return _fig_to_file(fig, output_path)
    return _fig_to_base64(fig)


def plot_per_class_accuracy(
    per_class: dict[str, dict],
    title: str = "Per-Class Accuracy",
    output_path: Path | None = None,
) -> str | None:
    """Create a grouped bar chart of producer's and user's accuracy per class.

    Args:
        per_class: Dict of {class_name: {producers_accuracy, users_accuracy, f1}}.
        title: Plot title.
        output_path: If provided, save to file; otherwise return base64.

    Returns:
        Base64 PNG string or filename.
    """
    if not _MPL_AVAILABLE:
        return None

    classes = list(per_class.keys())
    pa = [per_class[c].get("producers_accuracy", 0) for c in classes]
    ua = [per_class[c].get("users_accuracy", 0) for c in classes]
    f1 = [per_class[c].get("f1", 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 5))
    ax.bar(x - width, pa, width, label="Producer's Accuracy", color="#4A90D9")
    ax.bar(x, ua, width, label="User's Accuracy", color="#E8913A")
    ax.bar(x + width, f1, width, label="F1 Score", color="#5CB85C")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        return _fig_to_file(fig, output_path)
    return _fig_to_base64(fig)


def plot_model_comparison(
    model_metrics: dict[str, dict],
    title: str = "Model Comparison",
    output_path: Path | None = None,
) -> str | None:
    """Create a grouped bar chart comparing models across metrics.

    Args:
        model_metrics: {model_name: {overall_accuracy, kappa, f1_macro}}.
        title: Plot title.
        output_path: If provided, save to file; otherwise return base64.

    Returns:
        Base64 PNG string or filename.
    """
    if not _MPL_AVAILABLE:
        return None

    models = list(model_metrics.keys())
    metrics = ["overall_accuracy", "kappa", "f1_macro"]
    metric_labels = ["Overall Accuracy", "Kappa", "F1 (macro)"]
    colors = ["#4A90D9", "#E8913A", "#5CB85C"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [model_metrics[m].get(metric, 0) or 0 for m in models]
        ax.bar(x + i * width - width, values, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        return _fig_to_file(fig, output_path)
    return _fig_to_base64(fig)


def plot_error_density(
    rate_grid: list[list[float]],
    bounds: list[float],
    title: str = "Spatial Error Density",
    output_path: Path | None = None,
) -> str | None:
    """Create a heatmap of spatial error density.

    Args:
        rate_grid: 2D grid of error rates (-1 = no data).
        bounds: [west, south, east, north].
        title: Plot title.
        output_path: If provided, save to file; otherwise return base64.

    Returns:
        Base64 PNG string or filename.
    """
    if not _MPL_AVAILABLE:
        return None

    grid = np.array(rate_grid)
    masked = np.ma.masked_where(grid < 0, grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    im = ax.imshow(
        masked, cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=extent, origin="lower", aspect="auto",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Error Rate")

    if output_path:
        return _fig_to_file(fig, output_path)
    return _fig_to_base64(fig)


def generate_accuracy_report(
    results: dict[str, Any],
    output_path: str | Path,
    model_name: str = "Model",
    error_analysis: dict[str, Any] | None = None,
    error_density: dict[str, Any] | None = None,
) -> Path:
    """Generate an HTML accuracy assessment report.

    Args:
        results: Output from AccuracyAssessor.assess().
        output_path: Path for the HTML report.
        model_name: Model display name.
        error_analysis: Optional output from error analysis.
        error_density: Optional output from spatial error density.

    Returns:
        Path to generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plots = {}

    cm = results.get("confusion_matrix")
    labels = results.get("labels")
    if cm and labels:
        plots["confusion_matrix"] = plot_confusion_matrix(cm, labels, f"{model_name} - Confusion Matrix")

    per_class = results.get("per_class")
    if per_class:
        plots["per_class"] = plot_per_class_accuracy(per_class, f"{model_name} - Per-Class Accuracy")

    if error_density and "rate_grid" in error_density:
        plots["error_density"] = plot_error_density(
            error_density["rate_grid"], error_density["bounds"],
            f"{model_name} - Spatial Error Density",
        )

    # Build HTML
    html = _build_accuracy_html(results, plots, model_name, error_analysis)

    output_path.write_text(html)
    logger.info(f"Report saved to {output_path}")
    return output_path


def generate_comparison_report(
    comparison: dict[str, Any],
    output_path: str | Path,
    per_model_results: dict[str, dict] | None = None,
) -> Path:
    """Generate an HTML model comparison report.

    Args:
        comparison: Output from AccuracyAssessor.compare_models().
        output_path: Path for the HTML report.
        per_model_results: Optional per-model full results for detailed plots.

    Returns:
        Path to generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Model comparison chart
    models = comparison.get("models", {})
    valid_models = {k: v for k, v in models.items() if "error" not in v}
    if valid_models:
        plots["comparison"] = plot_model_comparison(valid_models, "Model Comparison")

    # Per-model confusion matrices
    if per_model_results:
        for model_name, result in per_model_results.items():
            cm = result.get("confusion_matrix")
            labels = result.get("labels")
            if cm and labels:
                key = f"cm_{model_name}"
                plots[key] = plot_confusion_matrix(cm, labels, f"{model_name} - Confusion Matrix")

    html = _build_comparison_html(comparison, plots)

    output_path.write_text(html)
    logger.info(f"Comparison report saved to {output_path}")
    return output_path


def _build_accuracy_html(
    results: dict[str, Any],
    plots: dict[str, str | None],
    model_name: str,
    error_analysis: dict[str, Any] | None = None,
) -> str:
    """Build the HTML content for an accuracy report."""

    oa = results.get("overall_accuracy", 0)
    kappa = results.get("kappa", 0)
    f1_m = results.get("f1_macro", 0)
    f1_w = results.get("f1_weighted", 0)
    n_pts = results.get("n_points", 0)

    sections = []

    # Summary metrics
    sections.append(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{oa:.4f}</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{kappa:.4f}</div>
            <div class="metric-label">Kappa</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{f1_m:.4f}</div>
            <div class="metric-label">F1 (macro)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{f1_w:.4f}</div>
            <div class="metric-label">F1 (weighted)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{n_pts}</div>
            <div class="metric-label">Reference Points</div>
        </div>
    </div>
    """)

    # Confusion matrix plot
    if plots.get("confusion_matrix"):
        sections.append(f"""
        <h2>Confusion Matrix</h2>
        <img src="data:image/png;base64,{plots['confusion_matrix']}" class="plot">
        """)

    # Per-class accuracy plot
    if plots.get("per_class"):
        sections.append(f"""
        <h2>Per-Class Accuracy</h2>
        <img src="data:image/png;base64,{plots['per_class']}" class="plot">
        """)

    # Per-class table
    per_class = results.get("per_class", {})
    if per_class:
        rows = ""
        for name, m in per_class.items():
            rows += f"""
            <tr>
                <td>{name}</td>
                <td>{m.get('producers_accuracy', 0):.4f}</td>
                <td>{m.get('users_accuracy', 0):.4f}</td>
                <td>{m.get('f1', 0):.4f}</td>
                <td>{m.get('support', 0)}</td>
            </tr>
            """
        sections.append(f"""
        <h2>Per-Class Metrics</h2>
        <table>
            <tr><th>Class</th><th>Producer's Acc</th><th>User's Acc</th><th>F1</th><th>Support</th></tr>
            {rows}
        </table>
        """)

    # Error analysis
    if error_analysis:
        top_conf = error_analysis.get("top_confusions", [])
        if top_conf:
            conf_rows = ""
            for pair, count in top_conf[:10]:
                conf_rows += f"<tr><td>{pair}</td><td>{count}</td></tr>"
            sections.append(f"""
            <h2>Top Confusion Pairs</h2>
            <table>
                <tr><th>True -> Predicted</th><th>Count</th></tr>
                {conf_rows}
            </table>
            """)

    # Error density plot
    if plots.get("error_density"):
        sections.append(f"""
        <h2>Spatial Error Distribution</h2>
        <img src="data:image/png;base64,{plots['error_density']}" class="plot">
        """)

    # Warnings
    warnings = results.get("warnings", [])
    if warnings:
        warn_items = "".join(f"<li>{w}</li>" for w in warnings)
        sections.append(f"""
        <h2>Warnings</h2>
        <ul class="warnings">{warn_items}</ul>
        """)

    body = "\n".join(sections)
    return _html_template(f"Accuracy Report: {model_name}", body)


def _build_comparison_html(
    comparison: dict[str, Any],
    plots: dict[str, str | None],
) -> str:
    """Build the HTML content for a comparison report."""

    sections = []

    # Comparison chart
    if plots.get("comparison"):
        sections.append(f"""
        <h2>Model Comparison</h2>
        <img src="data:image/png;base64,{plots['comparison']}" class="plot">
        """)

    # Summary table
    models = comparison.get("models", {})
    if models:
        rows = ""
        for name, m in models.items():
            if "error" in m:
                rows += f"<tr><td>{name}</td><td colspan='4'>Error: {m['error']}</td></tr>"
            else:
                oa = m.get("overall_accuracy")
                ka = m.get("kappa")
                f1 = m.get("f1_macro")
                oa_s = f"{oa:.4f}" if isinstance(oa, (int, float)) else "N/A"
                ka_s = f"{ka:.4f}" if isinstance(ka, (int, float)) else "N/A"
                f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{oa_s}</td>
                    <td>{ka_s}</td>
                    <td>{f1_s}</td>
                    <td>{m.get('n_points', 'N/A')}</td>
                </tr>
                """
        sections.append(f"""
        <h2>Model Summary</h2>
        <table>
            <tr><th>Model</th><th>Overall Accuracy</th><th>Kappa</th><th>F1 (macro)</th><th>Points</th></tr>
            {rows}
        </table>
        """)

    # Ranking
    ranking = comparison.get("ranking", [])
    if ranking:
        rank_rows = ""
        for i, r in enumerate(ranking, 1):
            rank_rows += f"<tr><td>{i}</td><td>{r['model']}</td><td>{r['overall_accuracy']:.4f}</td></tr>"
        sections.append(f"""
        <h2>Ranking</h2>
        <table>
            <tr><th>Rank</th><th>Model</th><th>Overall Accuracy</th></tr>
            {rank_rows}
        </table>
        <p>Best model: <strong>{comparison.get('best_model', 'N/A')}</strong></p>
        """)

    # Per-model confusion matrices
    cm_keys = [k for k in plots if k.startswith("cm_") and plots[k]]
    if cm_keys:
        sections.append("<h2>Per-Model Confusion Matrices</h2>")
        for key in cm_keys:
            model_name = key[3:]  # strip "cm_"
            sections.append(f"""
            <h3>{model_name}</h3>
            <img src="data:image/png;base64,{plots[key]}" class="plot">
            """)

    body = "\n".join(sections)
    return _html_template("Model Comparison Report", body)


def _html_template(title: str, body: str) -> str:
    """Wrap content in the HTML template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot {{
            max-width: 100%;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .warnings {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px 15px 15px 35px;
        }}
        .warnings li {{
            margin: 5px 0;
            color: #856404;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {body}
    <div class="footer">
        Generated by LCComparison2026
    </div>
</body>
</html>"""
