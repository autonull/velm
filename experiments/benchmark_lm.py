#!/usr/bin/env python3
"""Benchmark: Transformer vs VELM Lite vs VELM Full.

Compares architectures on configurable text datasets:
  - transformer: causal self-attention baseline (PyTorch, Adam)
  - velm_lite: simplified VELM — mean-pooling + MLP memory + linear head (PyTorch, Adam)
  - velm_full: canonical VELM — CALM + Miras + SWA + energy head (JAX, EGGROLL)

Usage:
  python experiments/benchmark_lm.py                                    # transformer + velm_lite, tiny_shakespeare
  python experiments/benchmark_lm.py --models transformer velm_lite velm_full
  python experiments/benchmark_lm.py --dataset tiny_stories
  python experiments/benchmark_lm.py --smoketest                        # quick validation
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.lib.datasets import get_dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Transformer baseline (PyTorch)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, d_model=64, nhead=4, nlayers=2, dim_feedforward=128, max_len=1024
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(nlayers)]
        )
        self.out = nn.Linear(d_model, vocab_size)
        self.adapter = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, inp, use_adapter=False):
        B, L = inp.shape
        x = self.embed(inp) + self.pos[:L].unsqueeze(0).to(inp.device)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=inp.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        if use_adapter:
            x = self.adapter(x)
        return self.out(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_batch(data, batch_size, seq_len, block_size=4):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    ix = (ix // block_size) * block_size
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pytorch(model, data, model_name, eval_iters, batch_size, seq_len, block_size, device):
    """Evaluate a PyTorch model (Transformer or VELM Lite)."""
    model.eval()
    losses, accs, cib_norms, latencies = [], [], [], []

    for _ in range(eval_iters):
        X, Y = get_batch(data, batch_size, seq_len, block_size)
        X, Y = X.to(device), Y.to(device)
        t0 = time.time()

        if model_name == "velm_lite":
            states, latents = model(X)
            B, N, _ = states.shape
            states_flat = states.view(B * N, -1)
            logits = model.decode_block_state(states_flat)
            targets = Y.view(B * N, block_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            acc = (
                (logits.view(-1, logits.size(-1)).argmax(-1) == targets.view(-1))
                .float()
                .mean()
                .item()
            )
            cib_norms.append(latents.norm(p=2, dim=2).mean().item())
        else:
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            acc = (logits.view(-1, logits.size(-1)).argmax(-1) == Y.view(-1)).float().mean().item()

        latencies.append(time.time() - t0)
        losses.append(loss.item())
        accs.append(acc)

    model.train()
    return (
        np.mean(losses),
        np.mean(accs),
        np.mean(cib_norms) if cib_norms else None,
        np.mean(latencies),
    )


def evaluate_jax_velm(velm_fn, params, data_np, eval_iters, batch_size, seq_len, block_size, key):
    """Evaluate the JAX VELM Full model on numpy data."""
    import jax

    losses, accs, latencies = [], [], []

    for _ in range(eval_iters):
        ix = np.random.randint(0, len(data_np) - seq_len, batch_size)
        ix = (ix // block_size) * block_size
        x = np.stack([data_np[i : i + seq_len] for i in ix])
        y = np.stack([data_np[i + 1 : i + seq_len + 1] for i in ix])

        t0 = time.time()
        loss, acc = velm_fn(params, x, y, key=key)
        latencies.append(time.time() - t0)
        losses.append(float(loss))
        accs.append(float(acc))
        key, _ = jax.random.split(key)

    return np.mean(losses), np.mean(accs), None, np.mean(latencies), key


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, model_order, args):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"VELM Benchmark — {args.dataset}", fontsize=16)

    colors = {"transformer": "#1f77b4", "velm_lite": "#ff7f0e", "velm_full": "#2ca02c"}

    for name in model_order:
        if name not in results:
            continue
        c = colors.get(name, "#999999")
        label = name.upper().replace("_", " ")
        r = results[name]
        n = len(r["train_loss"])
        steps = np.arange(n) * args.eval_interval

        axs[0, 0].plot(steps, r["train_loss"], label=label, color=c)
        axs[0, 1].plot(steps, r["val_loss"], label=label, color=c)
        axs[1, 0].plot(steps, r["val_acc"], label=label, color=c)
        axs[1, 1].plot(steps, [l * 1000 for l in r["val_latency"]], label=label, color=c)

        if r["throughput"]:
            t_steps = np.arange(1, len(r["throughput"]) + 1) * args.eval_interval
            axs[2, 0].plot(t_steps, r["throughput"], label=label, color=c)

        if r.get("cib_norm"):
            axs[2, 1].plot(steps, r["cib_norm"], label=label, color=c)

    axs[0, 0].set_title("Train Loss")
    axs[0, 0].legend()
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].legend()
    axs[1, 0].set_title("Validation Accuracy")
    axs[1, 0].legend()
    axs[1, 1].set_title("Inference Latency (ms)")
    axs[1, 1].legend()
    axs[2, 0].set_title("Training Throughput (tok/s)")
    axs[2, 0].legend()
    axs[2, 1].set_title("CIB Norm")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "composite_results.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer vs VELM Lite vs VELM Full")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["transformer", "velm_lite"],
        choices=["transformer", "velm_lite", "velm_full"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "shakespeare_full", "tiny_stories"],
        help="Dataset to use",
    )
    parser.add_argument("--out", type=str, default="results/benchmark")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoketest", action="store_true", help="Quick smoke test")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory for downloaded data"
    )
    parser.add_argument(
        "--velm_full_iters",
        type=int,
        default=None,
        help="Max iters for VELM Full (defaults to --max_iters)",
    )
    parser.add_argument(
        "--velm_full_pop",
        type=int,
        default=None,
        help="EGGROLL population size for VELM Full (auto-detects if not set)",
    )
    args = parser.parse_args()

    if args.smoketest:
        args.max_iters = 50
        args.eval_interval = 25
        args.eval_iters = 10
        args.batch_size = 16
        args.seq_len = 32
        # VELM Full is too slow for smoketest — skip it
        if "velm_full" in args.models:
            args.models = [m for m in args.models if m != "velm_full"]
            if not args.models:
                args.models = ["transformer", "velm_lite"]

    os.makedirs(args.out, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    train_data, val_data, vocab_size, stoi, itos = get_dataset(
        args.dataset, block_size=args.block_size, data_dir=args.data_dir
    )
    print(f"Dataset: {args.dataset}")
    print(f"Vocab size: {vocab_size}, Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Build models
    pytorch_models = {}
    model_info = {}

    if "transformer" in args.models:
        tf = TransformerLM(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            nlayers=5,
            dim_feedforward=256,
            max_len=args.seq_len,
        )
        pytorch_models["transformer"] = tf
        model_info["transformer"] = {"params": count_params(tf), "framework": "pytorch"}

    if "velm_lite" in args.models:
        from velm.lite import VelmFull

        vl = VelmFull(
            vocab_size=vocab_size,
            block_size=args.block_size,
            embed_dim=128,
            latent_dim=128,
            state_dim=128,
        )
        pytorch_models["velm_lite"] = vl
        model_info["velm_lite"] = {"params": count_params(vl), "framework": "pytorch"}

    # JAX VELM Full
    jax_velm_fn = None
    jax_params = None
    jax_key = None
    if "velm_full" in args.models:
        try:
            import jax
            import jax.numpy as jnp
            import equinox as eqx
            from velm.jax.model import VELM

            jax_key = jax.random.PRNGKey(SEED)
            vj = VELM(
                config_name="smoke",
                vocab_size=vocab_size,
                ae_hidden_dim=128,
                ae_ffn_intermediate=256,
                key=jax_key,
            )
            jax_params = eqx.filter(vj, eqx.is_array)
            param_count = sum(x.size for x in jax.tree.leaves(jax_params))
            model_info["velm_full"] = {"params": param_count, "framework": "jax"}

            # Store static (non-array) parts for reconstruction
            jax_static = eqx.filter(vj, lambda leaf: not eqx.is_array(leaf))

            @jax.jit
            def _eval_fn(params, x, y, key):
                model = eqx.combine(params, jax_static)
                return model.eval_batch(
                    params, jnp.array(x), jnp.array(y), key=key, chunk_size=args.block_size
                )

            jax_velm_fn = _eval_fn
            print(f"VELM Full loaded: {param_count} parameters (JAX)")
        except Exception as e:
            print(f"WARNING: Could not load VELM Full (JAX): {e}")
            print("Skipping VELM Full. Install JAX/Equinox to enable.")
            if "velm_full" in model_info:
                del model_info["velm_full"]

    for name, info in model_info.items():
        print(f"  {name}: {info['params']} params ({info['framework']})")

    # Results storage
    results = {}
    for name in model_info:
        results[name] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_latency": [],
            "throughput": [],
            "cib_norm": [],
        }

    # Train PyTorch models
    for name, model in pytorch_models.items():
        print(f"\n--- Training {name.upper()} ---")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        t_start_step = time.time()
        for iter_num in range(args.max_iters):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
                train_loss, train_acc, cib_norm, train_lat = evaluate_pytorch(
                    model,
                    train_data,
                    name,
                    args.eval_iters,
                    args.batch_size,
                    args.seq_len,
                    args.block_size,
                    device,
                )
                val_loss, val_acc, val_cib_norm, val_lat = evaluate_pytorch(
                    model,
                    val_data,
                    name,
                    args.eval_iters,
                    args.batch_size,
                    args.seq_len,
                    args.block_size,
                    device,
                )

                results[name]["train_loss"].append(train_loss)
                results[name]["val_loss"].append(val_loss)
                results[name]["val_acc"].append(val_acc)
                results[name]["val_latency"].append(val_lat)
                if cib_norm is not None:
                    results[name]["cib_norm"].append(val_cib_norm)

                t_end_step = time.time()
                steps_per_sec = (
                    args.eval_interval / (t_end_step - t_start_step) if iter_num > 0 else 0
                )
                throughput = steps_per_sec * args.batch_size * args.seq_len
                if iter_num > 0:
                    results[name]["throughput"].append(throughput)
                t_start_step = time.time()

                print(
                    f"Step {iter_num}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
                    + (f", CIB Norm {val_cib_norm:.4f}" if cib_norm is not None else "")
                    + (
                        f", Speed {steps_per_sec:.2f} steps/s, Throughput {throughput:.2f} tok/s"
                        if iter_num > 0
                        else ""
                    )
                )

            X, Y = get_batch(train_data, args.batch_size, args.seq_len, args.block_size)
            X, Y = X.to(device), Y.to(device)

            if name == "velm_lite":
                states, latents = model(X)
                B, N, _ = states.shape
                states_flat = states.view(B * N, -1)
                logits = model.decode_block_state(states_flat)
                targets = Y.view(B * N, args.block_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss + 1e-3 * latents.norm(p=2, dim=2).mean()
            else:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        t1 = time.time()
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        results[name]["peak_memory"] = peak_memory
        print(f"{name.upper()} Training Time: {t1 - t0:.2f}s")
        if peak_memory > 0:
            print(f"{name.upper()} Peak Memory: {peak_memory:.2f} MB")

    # Train JAX VELM Full (if included)
    if jax_velm_fn is not None:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        import optax

        print(f"\n--- Training VELM_FULL (JAX/gradient descent) ---")
        print("NOTE: VELM Full is designed for energy-based training with EGGROLL.")
        print("      CE-based gradient descent is suboptimal but enables architecture comparison.")
        print("      For proper training, use EGGROLL on GPU with --velm_full_pop >= 8")

        vf_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
        vf_opt_state = vf_opt.init(jax_params)
        vf_max_iters = args.velm_full_iters or args.max_iters

        train_np = np.array(train_data.numpy())
        val_np = np.array(val_data.numpy())

        @jax.jit
        def _train_step(params, opt_state, x, y, key):
            """One gradient step using CE loss (same objective as other models)."""

            def loss_fn(p):
                m = eqx.combine(p, jax_static)
                return m.eval_batch(p, x, y, key=key, chunk_size=args.block_size)

            (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, vf_opt_state = vf_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, acc

        t0 = time.time()
        t_start_step = time.time()
        for iter_num in range(vf_max_iters):
            jax_key, step_key = jax.random.split(jax_key)
            ix = np.random.randint(0, len(train_np) - args.seq_len, args.batch_size)
            ix = (ix // args.block_size) * args.block_size
            x = jnp.array(np.stack([train_np[i : i + args.seq_len] for i in ix]))
            y = jnp.array(np.stack([train_np[i + 1 : i + args.seq_len + 1] for i in ix]))

            jax_params, vf_opt_state, loss, acc = _train_step(
                jax_params, vf_opt_state, x, y, step_key
            )

            if iter_num % args.eval_interval == 0 or iter_num == vf_max_iters - 1:
                val_loss, val_acc, _, val_lat, jax_key = evaluate_jax_velm(
                    jax_velm_fn,
                    jax_params,
                    val_np,
                    args.eval_iters,
                    args.batch_size,
                    args.seq_len,
                    args.block_size,
                    jax_key,
                )
                train_loss, train_acc, _, _, jax_key = evaluate_jax_velm(
                    jax_velm_fn,
                    jax_params,
                    train_np,
                    args.eval_iters,
                    args.batch_size,
                    args.seq_len,
                    args.block_size,
                    jax_key,
                )

                results["velm_full"]["train_loss"].append(train_loss)
                results["velm_full"]["val_loss"].append(val_loss)
                results["velm_full"]["val_acc"].append(val_acc)
                results["velm_full"]["val_latency"].append(val_lat)

                t_end_step = time.time()
                steps_per_sec = (
                    args.eval_interval / (t_end_step - t_start_step) if iter_num > 0 else 0
                )
                throughput = steps_per_sec * args.batch_size * args.seq_len
                if iter_num > 0:
                    results["velm_full"]["throughput"].append(throughput)
                t_start_step = time.time()

                print(
                    f"Step {iter_num}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
                    f", Train CE {float(loss):.4f}"
                    + (
                        f", Speed {steps_per_sec:.2f} steps/s, Throughput {throughput:.2f} tok/s"
                        if iter_num > 0
                        else ""
                    )
                )

        t1 = time.time()
        results["velm_full"]["peak_memory"] = 0
        print(f"VELM_FULL Training Time: {t1 - t0:.2f}s")

    # Export CSV — handle different eval counts per model
    history_path = os.path.join(args.out, "history.csv")
    model_order = list(results.keys())
    if not model_order:
        print("No models to benchmark.")
        return

    # Find max number of eval points across all models
    max_evals = max(len(results[name]["train_loss"]) for name in model_order)
    steps = np.arange(max_evals) * args.eval_interval

    fieldnames = ["step"]
    for name in model_order:
        p = name.replace("_", "")
        fieldnames += [
            f"{p}_train_loss",
            f"{p}_val_loss",
            f"{p}_val_acc",
            f"{p}_val_latency",
            f"{p}_throughput",
        ]
        if name in ("velm_lite", "velm_full"):
            fieldnames.append(f"{p}_cib_norm")

    with open(history_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_evals):
            step = steps[i]
            row = {"step": step}
            for name in model_order:
                p = name.replace("_", "")
                n_evals = len(results[name]["train_loss"])
                if i < n_evals:
                    row[f"{p}_train_loss"] = results[name]["train_loss"][i]
                    row[f"{p}_val_loss"] = results[name]["val_loss"][i]
                    row[f"{p}_val_acc"] = results[name]["val_acc"][i]
                    row[f"{p}_val_latency"] = results[name]["val_latency"][i]
                    if name in ("velm_lite", "velm_full") and i < len(results[name]["cib_norm"]):
                        row[f"{p}_cib_norm"] = results[name]["cib_norm"][i]
                if i > 0 and (i - 1) < len(results[name]["throughput"]):
                    row[f"{p}_throughput"] = results[name]["throughput"][i - 1]
                else:
                    row[f"{p}_throughput"] = 0.0
            writer.writerow(row)
    print(f"History saved to {history_path}")

    # Plotting
    plot_results(results, model_order, args)

    # Summary
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Vocab size: {vocab_size}\n\n")
        for name in model_order:
            info = model_info.get(name, {})
            f.write(f"{name.upper()} params: {info.get('params', 'N/A')}\n")
            f.write(f"{name.upper()} Final Val Loss: {results[name]['val_loss'][-1]:.4f}\n")
            f.write(f"{name.upper()} Final Val Acc: {results[name]['val_acc'][-1]:.4f}\n")
            f.write(
                f"{name.upper()} Mean Val Latency: {np.mean(results[name]['val_latency']) * 1000:.2f} ms\n"
            )
            if results[name]["throughput"]:
                f.write(
                    f"{name.upper()} Mean Throughput: {np.mean(results[name]['throughput']):.2f} tok/s\n"
                )
            if results[name].get("peak_memory", 0) > 0:
                f.write(f"{name.upper()} Peak Memory: {results[name]['peak_memory']:.2f} MB\n")
            if results[name]["cib_norm"]:
                f.write(f"{name.upper()} Final CIB Norm: {results[name]['cib_norm'][-1]:.4f}\n")
            f.write("\n")
    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
