#!/usr/bin/env python3
"""
VELM Comprehensive Reporting Script

Parses the benchmark history and summary files to generate a highly detailed,
composite evaluation report (velm_comprehensive_report.png) matching the
academic-scale rigor required by the project.

It visualizes:
  - Validation Loss vs Step
  - Validation Accuracy vs Step
  - Inference Latency vs Step
  - Throughput vs Step
  - Iteration Time vs Step
  - CIB Norm vs Step (if applicable)
  - Also displays a text box with Model Parameter Counts and Vocab Size

Usage:
  PYTHONPATH=src python src/generate_comprehensive_report.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_report(results_dir="results/benchmark"):
    history_path = os.path.join(results_dir, "history.csv")
    summary_path = os.path.join(results_dir, "summary.txt")

    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found. Run benchmark_lm.py first.")
        return

    # Parse History
    data = {"step": []}
    models = set()
    metrics_list = ["train_loss", "val_loss", "val_acc", "val_latency", "throughput", "iteration_time", "cib_norm", "params"]
    with open(history_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for field in fieldnames:
            if field == "step":
                continue
            for metric in metrics_list:
                if field.endswith(f"_{metric}"):
                    model_name = field[:-(len(metric)+1)]
                    models.add(model_name)
                    break

        for m in models:
            data[m] = { metric: [] for metric in metrics_list }

        for row in reader:
            data["step"].append(float(row["step"]))
            for m in models:
                for metric in data[m]:
                    key = f"{m}_{metric}"
                    if key in row and row[key] != "":
                        try:
                            data[m][metric].append(float(row[key]))
                        except ValueError:
                            data[m][metric].append(np.nan)
                    else:
                        data[m][metric].append(np.nan)

    # Parse Summary for extra text info
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()

    # Create composite plot
    # We want a 3x2 grid of plots, plus maybe text box. We can make a 4x2 grid
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("VELM Comprehensive Evaluation Report", fontsize=20, fontweight='bold', y=0.98)

    colors = {"transformer": "#1f77b4", "velmlite": "#ff7f0e", "velmfull": "#2ca02c"}

    steps = np.array(data["step"])

    metrics = [
        ("val_loss", "Validation Loss", axs[0, 0]),
        ("val_acc", "Validation Accuracy", axs[0, 1]),
        ("val_latency", "Inference Latency (s)", axs[1, 0]),
        ("throughput", "Throughput (tokens/s)", axs[1, 1]),
        ("iteration_time", "Iteration Time (s/step)", axs[2, 0]),
        ("cib_norm", "CIB Latent Norm", axs[2, 1]),
    ]

    for m in models:
        c = colors.get(m.lower(), "#999999")
        label = m.upper()

        for metric_key, title, ax in metrics:
            if data[m].get(metric_key) and len(data[m][metric_key]) == len(steps):
                y_data = np.array(data[m][metric_key])
                # Filter out zeroes for throughput/iteration time if any
                valid = y_data > 0 if metric_key in ("throughput", "iteration_time") else np.ones_like(y_data, dtype=bool)
                ax.plot(steps[valid], y_data[valid], label=label, color=c, linewidth=2)

    for metric_key, title, ax in metrics:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Training Steps")
        ax.grid(True, linestyle='--', alpha=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")

    # Bottom row for extra plots or text
    ax_text1 = axs[3, 0]
    ax_text2 = axs[3, 1]

    ax_text1.axis('off')
    ax_text2.axis('off')

    # Train loss on bottom left
    for m in models:
        c = colors.get(m.lower(), "#999999")
        if data[m].get("train_loss") and len(data[m]["train_loss"]) == len(steps):
            ax_text1.plot(steps, data[m]["train_loss"], label=m.upper(), color=c, linewidth=2)
    ax_text1.axis('on')
    ax_text1.set_title("Training Loss", fontsize=14)
    ax_text1.set_xlabel("Training Steps")
    ax_text1.grid(True, linestyle='--', alpha=0.7)
    if ax_text1.get_legend_handles_labels()[0]:
        ax_text1.legend(loc="best")

    # Text summary on bottom right
    ax_text2.text(0.05, 0.95, "Evaluation Telemetry summary",
                  fontsize=14, fontweight='bold', va='top')

    if summary_text:
        ax_text2.text(0.05, 0.85, summary_text,
                      fontsize=12, va='top', family='monospace',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    else:
        info = ""
        for m in models:
            p = data[m]["params"][0] if data[m]["params"] else "N/A"
            info += f"{m.upper()} Params: {p}\n"
        ax_text2.text(0.05, 0.85, info,
                      fontsize=12, va='top', family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    out_path = os.path.join(results_dir, "velm_comprehensive_report.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comprehensive report saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/benchmark", help="Directory containing benchmark history.csv")
    args = parser.parse_args()
    generate_report(args.results_dir)
