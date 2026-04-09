# VELM vs Vanilla Transformer Benchmark Report

This document details the comparative benchmarking of the Vector-Evolution Language Model (VELM) architectural variants against a Vanilla Transformer baseline on the `tiny_shakespeare` dataset. The benchmark rigorously bounds all models to a similar parameter count (~700K) to objectively assess architectural efficiency and capacity. Evaluations were run using 500 steps to shorten training time while remaining indicative of relative performance.

## Executive Summary

**VELM Lite** significantly outperforms the Vanilla Transformer baseline across all key metrics:
- **Loss** was slashed significantly (0.5043 vs 1.9742)
- **Accuracy** more than doubled (85.91% vs 41.46%)
- **Latency** improved by nearly ~50% (65.77 ms vs 128.65 ms)
- **Throughput** surged dramatically (44.7k tok/s vs 10.5k tok/s)
- Achieved **despite having ~12% fewer parameters** (626k vs 712k).

The **VELM Hybrid (Deep Variant)** also demonstrates trading inference speed for deeper reasoning (using Miras memory blocks and SWA layers), effectively outpacing the baseline Transformer in learning capacity (Accuracy: 77.20% vs 41.46%) within the constrained parameter budget, while taking longer to process (127.69 ms). The Fast Variant shows intermediate performance and learning curve compared to Deep.

---

## Detailed Model Metrics

*Dataset: `tiny_shakespeare` | Vocabulary Size: `65`*

### 1. Vanilla Transformer (Baseline)
- **Parameters**: 712,001
- **Final Validation Loss**: 1.9742
- **Final Validation Accuracy**: 41.46%
- **Mean Validation Latency**: 128.65 ms
- **Mean Throughput**: 10,544.45 tok/s

### 2. VELM Lite (Best Overall Performance)
- **Parameters**: 626,848
- **Final Validation Loss**: 0.5043
- **Final Validation Accuracy**: 85.91%
- **Mean Validation Latency**: 65.77 ms
- **Mean Throughput**: 44,723.92 tok/s
- **Final CIB Norm**: 6.6167

### 3. VELM Hybrid (Fast Variant)
- **Parameters**: 726,950
- **Final Validation Loss**: 2.0088
- **Final Validation Accuracy**: 47.71%
- **Mean Validation Latency**: 134.85 ms
- **Mean Throughput**: 14,206.38 tok/s
- **Final CIB Norm**: 12.5275

### 4. VELM Hybrid (Deep Variant)
- **Parameters**: 716,064
- **Final Validation Loss**: 0.8412
- **Final Validation Accuracy**: 77.20%
- **Mean Validation Latency**: 127.69 ms
- **Mean Throughput**: 13,252.10 tok/s
- **Final CIB Norm**: 6.1211

---

## Visualizations

The composite benchmark charts containing the time-series evaluations (loss/accuracy vs. time/step), parameter distributions, and throughput analysis can be found at:

- `results/benchmark/velm_comprehensive_report.png`
- `results/benchmark/composite_results.png`

*(Raw telemetry stored in `results/benchmark/history.csv` and `results/benchmark/summary.txt`)*

## Methodology

Evaluations were conducted natively utilizing the PyTorch ecosystem proxies (`experiments/benchmark_lm.py`). The parameter constraints were strictly enforced by dynamically adjusting architectural parameters such as the `state_dim` (compensating for new components like Latent Thoughts while honoring divisibility rules for Sliding Window Attention) to ensure the competing baselines were never disadvantaged by size. Due to time constraints, the benchmark was abridged to 500 iterations rather than the default of 5000.
