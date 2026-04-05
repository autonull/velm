# VELM: Vector-Evolution Language Model

## A Self-Evolving, Continuous-Latent Language Model Architecture

**Status:** Research — preliminary benchmarks complete, Full model training in progress

### Abstract

VELM proposes a composite language model architecture that integrates six recent
advances into a unified system: continuous next-vector prediction (CALM), deep
nonlinear associative memory (Miras), gradient-free evolution strategies at scale
(EGGROLL), query-only test-time training for long context (qTTT), reasoning
compression via conditional information bottleneck (CIB), and group-based
open-ended self-improvement (GEA).

The key insight is that these innovations are not merely additive — they unlock
capabilities impossible under any single paradigm:

- **Nonlinear RNNs become trainable** because EGGROLL eliminates the need for
  backpropagation through time
- **Continuous-latent generation** becomes efficient because CALM compresses K
  tokens into single vectors, reducing autoregressive steps by K×
- **Deep memory** over continuous vectors replaces shallow linear recurrence with
  MLP-based associative memory that can actually track state
- **Long-context failures** are fixed at inference time via query-only TTT,
  without retraining or growing the KV cache
- **Reasoning efficiency** is enforced structurally via CIB, compressing
  chain-of-thought without accuracy loss
- **The entire system self-improves** through group evolution: populations of
  models share experience and evolve weights + workflows without human intervention

### Two Implementations

VELM is provided in two implementations that share the same high-level API
and are drop-in replacements for each other:

| | **VELM Full** | **VELM Lite** |
|---|---|---|
| **Framework** | JAX / Equinox | PyTorch |
| **Compression** | CALM autoencoder (VAE-style) | Block mean-pooling |
| **Memory** | Miras deep associative memory + SWA | Simple MLP memory |
| **Decoding** | Energy-based generative head | Linear projection |
| **Training** | EGGROLL (gradient-free ES) | Adam |
| **Purpose** | Canonical research implementation | Fast iteration, prototyping, benchmarking |
| **Params (tiny)** | ~817K | ~709K |

Both implement the same architectural prior:
```
tokens → block compression → continuous latent memory → block decoding → tokens
```

### Preliminary Results

Benchmarked on Tiny Shakespeare (~1.1M chars, 65-char vocabulary), strict
parameter count matching (~700K each), 2000 training steps:

| Metric | Transformer | VELM Lite | Delta |
|---|---|---|---|
| **Val Accuracy** | 49.7% | **86.3%** | **+36.6pp** |
| **Val Loss** | 2.002 | **0.529** | **-73%** |
| **Peak Memory** | 88.5 MB | **46.5 MB** | **-47%** |
| **Throughput** | 134K tok/s | 50K tok/s | -63% |
| **Train Time** | 30s | 81s | +2.7× |

VELM Lite achieves near-doubled accuracy with half the memory at matched
parameter count. The throughput penalty is expected — block-level autoregression
cannot be parallelized like self-attention.

**VELM Full** requires GPU + EGGROLL training (gradient-free evolution strategy).
The energy-based head cannot be trained with standard backpropagation. See
`experiments/train_velm_full_proper.py` for the proper two-phase training pipeline.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ REPRESENTATION: CALM Autoencoder → Miras Deep Memory → EBM  │
│ K tokens → 1 continuous vector → MLP memory backbone → head  │
├─────────────────────────────────────────────────────────────┤
│ TRAINING: EGGROLL (gradient-free, int8-native) + CIB loss    │
│ Low-rank ES populations, fitness = quality × compression     │
├─────────────────────────────────────────────────────────────┤
│ INFERENCE: qTTT query adaptation + CIB budget control        │
│ Frozen KV cache, adaptive query projections, efficient CoT   │
├─────────────────────────────────────────────────────────────┤
│ SELF-IMPROVEMENT: GEA group evolution + EGGROLL populations  │
│ Experience sharing across model variants, open-ended search  │
└─────────────────────────────────────────────────────────────┘
```

### Source Papers

| Paper | Contribution | Key Innovation |
|-------|-------------|----------------|
| CALM (Shao et al., 2025) | Representation | Compress K tokens → 1 continuous vector, energy-based decode |
| Miras (Behrouz et al., 2025) | Backbone | Unified framework: 4-choice deep associative memory |
| EGGROLL (Sarkar et al., 2025) | Training | Gradient-free ES at billion-param scale, int8 native |
| qTTT (Bansal et al., 2025) | Inference | Query-only TTT fixes score dilution in long context |
| CIB (2025) | Efficiency | Reasoning as compression via conditional info bottleneck |
| GEA (Weng et al., 2025) | Self-improvement | Group-evolving agents with experience sharing |

### Project Structure

```
VELM/
├── README.md
├── FIX.md                        # Known issues to address
├── docs/
│   ├── paper_outline.md
│   ├── architecture.md
│   ├── synergies.md
│   ├── experiments.md
│   └── figures/
├── src/velm/
│   ├── jax/                      # VELM Full — JAX/Equinox (canonical)
│   │   ├── model/                # CALM, Miras, energy head, full VELM
│   │   ├── training/             # EGGROLL optimizer + fitness
│   │   ├── inference/            # qTTT + CIB budget
│   │   └── evolution/            # GEA (fixed + multi-island)
│   └── lite/                     # VELM Lite — PyTorch (fast iteration)
├── experiments/
│   ├── lib/                      # Shared: datasets, models, plots
│   ├── run_all.py                # Turnkey runner (--smoketest)
│   ├── benchmark_lm.py           # Transformer vs VELM Lite
│   ├── train_velm_full_proper.py # VELM Full two-phase training
│   ├── velm_vs_transformer.py    # Synthetic task comparison
│   └── qttt_demo.py              # qTTT adaptation demo
├── tests/
└── results/                      # Experiment outputs (gitignored)
```

### Getting Started

**Quick start — run all experiments in ~30 seconds:**
```bash
python experiments/run_all.py --smoketest
```

**Benchmark Transformer vs VELM Lite:**
```bash
python experiments/benchmark_lm.py                              # Tiny Shakespeare, default settings
python experiments/benchmark_lm.py --dataset tiny_stories       # Different dataset
python experiments/benchmark_lm.py --smoketest                  # Quick validation
```

**Train VELM Full (requires GPU):**
```bash
# Phase 1: Pretrain autoencoder
python experiments/train_velm_full_proper.py --phase 1 --ae-steps 500

# Phase 2: Train backbone + energy head with EGGROLL
python experiments/train_velm_full_proper.py --phase 2 --ae-ckpt results/velm_full_train/ae_final.jax
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

### Known Issues

See [`FIX.md`](FIX.md) for tracked issues and their status. The most critical
(GEA/EGGROLL interface mismatch) has been fixed.
