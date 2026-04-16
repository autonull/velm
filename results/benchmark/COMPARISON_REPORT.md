# VELM vs Vanilla Transformer Benchmark Report

This document details the comparative benchmarking of the Vector-Evolution Language Model (VELM) architectural variants against a Vanilla Transformer baseline on the `tiny_shakespeare` dataset. The benchmark rigorously bounds all models to a similar parameter count (~700K) to objectively assess architectural efficiency and capacity.

## Executive Summary

**VELM Lite** significantly outperforms the Vanilla Transformer baseline across all key metrics:
- **Loss** was slashed by nearly 73% (0.5422 vs 1.9990)
- **Accuracy** more than doubled (84.82% vs 40.84%)
- **Latency** improved by ~37% (20.40 ms vs 32.72 ms)
- **Throughput** surged dramatically (20k tok/s vs 7.8k tok/s)
- Achieved **despite having ~11% fewer parameters** (626k vs 703k).

The **VELM Hybrid** variants successfully demonstrated trading inference speed for deeper reasoning (using Miras memory blocks and SWA layers), both comfortably outpacing the baseline Transformer in learning capacity within the constrained parameter budget.

---

## Detailed Model Metrics

*Dataset: `tiny_shakespeare` | Vocabulary Size: `65`*

### 1. Vanilla Transformer (Baseline)
- **Parameters**: 703,809
- **Final Validation Loss**: 1.9990
- **Final Validation Accuracy**: 40.84%
- **Mean Validation Latency**: 32.72 ms
- **Mean Throughput**: 7,872.56 tok/s

### 2. VELM Lite (Best Overall Performance)
- **Parameters**: 626,848
- **Final Validation Loss**: 0.5422
- **Final Validation Accuracy**: 84.82%
- **Mean Validation Latency**: 20.40 ms
- **Mean Throughput**: 20,046.71 tok/s
- **Final CIB Norm**: 9.1655

### 3. VELM Hybrid (Fast Variant)
- **Parameters**: 726,950
- **Final Validation Loss**: 1.2283
- **Final Validation Accuracy**: 66.62%
- **Mean Validation Latency**: 52.56 ms
- **Mean Throughput**: 7,035.46 tok/s
- **Final CIB Norm**: 7.7749

### 4. VELM Hybrid (Deep Variant)
- **Parameters**: 716,064
- **Final Validation Loss**: 0.8891
- **Final Validation Accuracy**: 76.22%
- **Mean Validation Latency**: 69.14 ms
- **Mean Throughput**: 5,085.72 tok/s
- **Final CIB Norm**: 3.1221

---

## Visualizations

The composite benchmark charts containing the time-series evaluations (loss/accuracy vs. time/step), parameter distributions, and throughput analysis can be found at:

- `results/benchmark/velm_comprehensive_report.png`
- `results/benchmark/composite_results.png`

*(Raw telemetry stored in `results/benchmark/history.csv` and `results/benchmark/summary.txt`)*

## Methodology

Evaluations were conducted natively utilizing the PyTorch ecosystem proxies (`experiments/benchmark_lm.py`). The parameter constraints were strictly enforced by dynamically adjusting architectural parameters such as the `state_dim` (compensating for new components like Latent Thoughts while honoring divisibility rules for Sliding Window Attention) to ensure the competing baselines were never disadvantaged by size.
