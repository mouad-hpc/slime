#!/usr/bin/env python3
"""
LoRA vs Full Fine-Tuning Benchmark Report
==========================================
Fetches metrics from two MLflow runs and generates a self-contained HTML comparison report.

Usage:
    python tools/benchmark_report.py \
        --experiment slime-lora-benchmark \
        --lora-run lora-r32-benchmark \
        --baseline-run full-ft-benchmark \
        --output benchmark_report.html

    # Upload the report back to MLflow as an artifact:
    python tools/benchmark_report.py \
        --experiment slime-lora-benchmark \
        --lora-run lora-r32-benchmark \
        --baseline-run full-ft-benchmark \
        --upload-artifact
"""

import argparse
import base64
import io
import sys
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("ERROR: mlflow is required. Install with: pip install mlflow", file=sys.stderr)
    sys.exit(1)


COLORS = {"lora": "#2196F3", "baseline": "#FF5722"}


def find_run(client, experiment_id, run_name):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"',
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"ERROR: No run found with name '{run_name}'", file=sys.stderr)
        sys.exit(1)
    return runs[0]


def get_metric_history(client, run_id, key):
    try:
        history = client.get_metric_history(run_id, key)
        return [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)]
    except Exception:
        return []


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_line_chart(title, metric_key, lora_data, baseline_data, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    if lora_data:
        steps, vals = zip(*lora_data)
        ax.plot(steps, vals, color=COLORS["lora"], label="LoRA", linewidth=1.5, marker=".", markersize=4)
    if baseline_data:
        steps, vals = zip(*baseline_data)
        ax.plot(steps, vals, color=COLORS["baseline"], label="Full FT", linewidth=1.5, marker=".", markersize=4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or metric_key)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def make_bar_chart(title, lora_vals, baseline_vals, ylabel):
    fig, ax = plt.subplots(figsize=(5, 4))
    lora_avg = sum(lora_vals) / len(lora_vals) if lora_vals else 0
    baseline_avg = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0
    bars = ax.bar(["LoRA", "Full FT"], [lora_avg, baseline_avg],
                  color=[COLORS["lora"], COLORS["baseline"]], width=0.5)
    for bar, val in zip(bars, [lora_avg, baseline_avg]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    return fig_to_base64(fig)


def safe_avg(data):
    vals = [v for _, v in data] if data else []
    return sum(vals) / len(vals) if vals else 0


def safe_max(data):
    vals = [v for _, v in data] if data else []
    return max(vals) if vals else 0


def build_html(charts, summary, lora_tags, baseline_tags):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def tag(key, tags):
        return tags.get(key, "N/A")

    rows = ""
    for label, key, fmt in summary:
        rows += f"<tr><td>{label}</td><td>{fmt(key, 'lora')}</td><td>{fmt(key, 'baseline')}</td></tr>\n"

    chart_html = ""
    for title, b64 in charts:
        chart_html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{b64}" alt="{title}">
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LoRA vs Full FT Benchmark</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 8px; }}
  h2 {{ color: #555; margin-top: 32px; }}
  .meta {{ color: #888; font-size: 0.85em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: right; }}
  th {{ background: #f5f5f5; text-align: left; }}
  td:first-child {{ text-align: left; font-weight: 500; }}
  .chart {{ margin: 24px 0; text-align: center; }}
  .chart img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>LoRA vs Full Fine-Tuning Benchmark</h1>
<p class="meta">Generated {timestamp}</p>
<p class="meta">
  LoRA: rank={tag("lora.rank", lora_tags)}, alpha={tag("lora.alpha", lora_tags)}, targets={tag("lora.target_modules", lora_tags)}<br>
  Model: {tag("model.name", lora_tags)}
</p>

<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>LoRA</th><th>Full FT</th></tr>
{rows}
</table>

<h2>Charts</h2>
<div class="grid">
{chart_html}
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate LoRA vs Full FT benchmark report from MLflow")
    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument("--lora-run", required=True, help="MLflow run name for LoRA run")
    parser.add_argument("--baseline-run", required=True, help="MLflow run name for baseline run")
    parser.add_argument("--output", default="benchmark_report.html", help="Output HTML file")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default: env or local)")
    parser.add_argument("--upload-artifact", action="store_true", help="Upload report as MLflow artifact to the LoRA run")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"ERROR: Experiment '{args.experiment}' not found", file=sys.stderr)
        sys.exit(1)

    lora_run = find_run(client, experiment.experiment_id, args.lora_run)
    baseline_run = find_run(client, experiment.experiment_id, args.baseline_run)

    print(f"LoRA run:     {lora_run.info.run_id} ({args.lora_run})")
    print(f"Baseline run: {baseline_run.info.run_id} ({args.baseline_run})")

    lora_tags = lora_run.data.tags
    baseline_tags = baseline_run.data.tags

    # Fetch metric histories
    metric_keys = [
        "perf/step_time",
        "perf/actor_train_tok_per_s",
        "perf/actor_train_tflops",
        "memory/peak_allocated_gb",
        "memory/current_allocated_gb",
        "rollout/reward",
        "train/grad_norm",
    ]

    lora_metrics = {}
    baseline_metrics = {}
    for key in metric_keys:
        lora_metrics[key] = get_metric_history(client, lora_run.info.run_id, key)
        baseline_metrics[key] = get_metric_history(client, baseline_run.info.run_id, key)

    # Build charts
    charts = []

    charts.append(("Step Time (s)", make_line_chart(
        "Step Time", "perf/step_time",
        lora_metrics["perf/step_time"], baseline_metrics["perf/step_time"],
        ylabel="Seconds")))

    charts.append(("GPU Memory (Peak Allocated GB)", make_line_chart(
        "GPU Memory (Peak Allocated)", "memory/peak_allocated_gb",
        lora_metrics["memory/peak_allocated_gb"], baseline_metrics["memory/peak_allocated_gb"],
        ylabel="GB")))

    charts.append(("Throughput (tok/s)", make_bar_chart(
        "Training Throughput",
        [v for _, v in lora_metrics["perf/actor_train_tok_per_s"]],
        [v for _, v in baseline_metrics["perf/actor_train_tok_per_s"]],
        ylabel="Tokens/s")))

    charts.append(("TFLOPS", make_bar_chart(
        "Training TFLOPS",
        [v for _, v in lora_metrics["perf/actor_train_tflops"]],
        [v for _, v in baseline_metrics["perf/actor_train_tflops"]],
        ylabel="TFLOPS")))

    charts.append(("Reward Curve", make_line_chart(
        "Reward", "rollout/reward",
        lora_metrics["rollout/reward"], baseline_metrics["rollout/reward"],
        ylabel="Reward")))

    charts.append(("Gradient Norm", make_line_chart(
        "Gradient Norm", "train/grad_norm",
        lora_metrics["train/grad_norm"], baseline_metrics["train/grad_norm"],
        ylabel="Grad Norm")))

    # Build summary table
    lora_total = lora_tags.get("model.total_params", "?")
    lora_trainable = lora_tags.get("model.trainable_params", "?")
    lora_ratio = lora_tags.get("model.trainable_ratio", "?")
    bl_total = baseline_tags.get("model.total_params", "?")
    bl_trainable = baseline_tags.get("model.trainable_params", "?")
    bl_ratio = baseline_tags.get("model.trainable_ratio", "?")

    lora_step_avg = safe_avg(lora_metrics["perf/step_time"])
    bl_step_avg = safe_avg(baseline_metrics["perf/step_time"])
    lora_mem_peak = safe_max(lora_metrics["memory/peak_allocated_gb"])
    bl_mem_peak = safe_max(baseline_metrics["memory/peak_allocated_gb"])
    lora_tok_avg = safe_avg(lora_metrics["perf/actor_train_tok_per_s"])
    bl_tok_avg = safe_avg(baseline_metrics["perf/actor_train_tok_per_s"])

    summary_data = {
        "lora": {
            "total_params": lora_total, "trainable_params": lora_trainable,
            "trainable_ratio": lora_ratio, "avg_step_time": f"{lora_step_avg:.2f}s",
            "peak_memory": f"{lora_mem_peak:.2f} GB", "avg_throughput": f"{lora_tok_avg:.1f} tok/s",
        },
        "baseline": {
            "total_params": bl_total, "trainable_params": bl_trainable,
            "trainable_ratio": bl_ratio, "avg_step_time": f"{bl_step_avg:.2f}s",
            "peak_memory": f"{bl_mem_peak:.2f} GB", "avg_throughput": f"{bl_tok_avg:.1f} tok/s",
        },
    }

    summary_rows = [
        ("Total Parameters", "total_params", lambda k, r: summary_data[r][k]),
        ("Trainable Parameters", "trainable_params", lambda k, r: summary_data[r][k]),
        ("Trainable Ratio", "trainable_ratio", lambda k, r: summary_data[r][k]),
        ("Avg Step Time", "avg_step_time", lambda k, r: summary_data[r][k]),
        ("Peak GPU Memory", "peak_memory", lambda k, r: summary_data[r][k]),
        ("Avg Throughput", "avg_throughput", lambda k, r: summary_data[r][k]),
    ]

    html = build_html(charts, summary_rows, lora_tags, baseline_tags)

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report written to {args.output}")

    if args.upload_artifact:
        with mlflow.start_run(run_id=lora_run.info.run_id):
            mlflow.log_artifact(args.output, artifact_path="benchmark")
        print(f"Report uploaded as artifact to run {lora_run.info.run_id}")


if __name__ == "__main__":
    main()
