#!/usr/bin/env python3
"""
Generate comprehensive evaluation reports for VELM benchmarks.
Reads the `results/benchmark/history.csv` and `summary.txt` and creates
composite charts for academic-scale reporting.
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

def generate_report(results_dir, out_file="velm_comprehensive_report.png"):
    history_file = os.path.join(results_dir, "history.csv")
    summary_file = os.path.join(results_dir, "summary.txt")

    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        sys.exit(1)

    steps = []
    data = {}

    with open(history_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            for key, val in row.items():
                if key != 'step':
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(val))
                    except ValueError:
                        data[key].append(np.nan)

    models = set()
    for key in data.keys():
        model_name = key.split('_')[0]
        models.add(model_name)

    # Read summary for params and latency
    params = {}
    latencies = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            for line in f:
                for model in ["TRANSFORMER", "VELM_LITE", "VELM_HYBRID"]:
                    if line.startswith(f"{model} params:"):
                        params[model.lower()] = int(line.split(':')[1].strip())
                    if line.startswith(f"{model} Mean Val Latency:"):
                        latencies[model.lower()] = float(line.split(':')[1].split('ms')[0].strip())

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VELM vs Baseline: Comprehensive Evaluation", fontsize=18)

    colors = {
        'transformer': '#1f77b4',
        'velmlite': '#ff7f0e',
        'velmhybrid': '#2ca02c'
    }

    for model in models:
        c = colors.get(model, 'k')
        label = model.upper()

        # Train Loss
        train_key = f"{model}_train_loss"
        if train_key in data:
            axs[0, 0].plot(steps, data[train_key], label=label, color=c)

        # Val Loss
        val_key = f"{model}_val_loss"
        if val_key in data:
            axs[0, 1].plot(steps, data[val_key], label=label, color=c)

        # Val Acc
        acc_key = f"{model}_val_acc"
        if acc_key in data:
            axs[0, 2].plot(steps, data[acc_key], label=label, color=c)

        # Throughput
        tp_key = f"{model}_throughput"
        if tp_key in data:
            axs[1, 0].plot(steps, data[tp_key], label=label, color=c)

    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Steps")
    axs[0, 0].set_ylabel("Cross Entropy")
    axs[0, 0].legend()

    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_xlabel("Steps")
    axs[0, 1].set_ylabel("Cross Entropy")
    axs[0, 1].legend()

    axs[0, 2].set_title("Validation Accuracy")
    axs[0, 2].set_xlabel("Steps")
    axs[0, 2].set_ylabel("Accuracy")
    axs[0, 2].legend()

    axs[1, 0].set_title("Throughput")
    axs[1, 0].set_xlabel("Steps")
    axs[1, 0].set_ylabel("Tokens / sec")
    axs[1, 0].legend()

    # Param counts
    names = list(params.keys())
    counts = [params[n] for n in names]
    bar_colors = [colors.get(n.replace('_', ''), 'k') for n in names]
    axs[1, 1].bar(names, counts, color=bar_colors)
    axs[1, 1].set_title("Parameter Count")
    axs[1, 1].set_ylabel("Params")

    # Latency
    l_names = list(latencies.keys())
    l_vals = [latencies[n] for n in l_names]
    l_colors = [colors.get(n.replace('_', ''), 'k') for n in l_names]
    axs[1, 2].bar(l_names, l_vals, color=l_colors)
    axs[1, 2].set_title("Mean Inference Latency")
    axs[1, 2].set_ylabel("Latency (ms)")

    plt.tight_layout()
    out_path = os.path.join(results_dir, out_file)
    plt.savefig(out_path, dpi=300)
    print(f"Comprehensive report saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/benchmark", help="Directory with history.csv")
    parser.add_argument("--out", default="velm_comprehensive_report.png", help="Output filename")
    args = parser.parse_args()
    generate_report(args.dir, args.out)
