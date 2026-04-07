#!/usr/bin/env python3
"""Benchmark: Transformer vs VELM Lite.

Compares architectures on configurable text datasets:
  - transformer: causal self-attention baseline (PyTorch, Adam)
  - velm_lite: simplified VELM — mean-pooling + MLP memory + linear head (PyTorch, Adam)

VELM Full (JAX/EGGROLL) is trained separately via experiments/train_velm_full_proper.py.

Usage:
  python experiments/benchmark_lm.py                                    # both models, tiny_shakespeare
  python experiments/benchmark_lm.py --dataset tiny_stories
  python experiments/benchmark_lm.py --smoketest                        # quick validation
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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
# Transformer baseline
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
def evaluate(model, data, model_name, eval_iters, batch_size, seq_len, block_size, device):
    model.eval()
    losses, accs, cib_norms, latencies = [], [], [], []

    for _ in range(eval_iters):
        X, Y = get_batch(data, batch_size, seq_len, block_size)
        X, Y = X.to(device), Y.to(device)
        t0 = time.time()

        if model_name.startswith("velm"):
            states, latents = model(X)
            B, N, _ = states.shape
            states_flat = states.view(B * N, -1)
            logits = model.decode_block_state(states_flat)

            logits = logits.view(B, N * block_size, -1)
            logits = logits[:, :Y.size(1), :]

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.view(-1))
            acc = (
                (logits.reshape(-1, logits.size(-1)).argmax(-1) == Y.view(-1))
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results, args):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"VELM Benchmark — {args.dataset}", fontsize=16)

    colors = {"transformer": "#1f77b4", "velm_lite": "#ff7f0e"}

    for name in results:
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
    parser = argparse.ArgumentParser(description="Benchmark Transformer vs VELM Lite")
    parser.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "shakespeare_full", "tiny_stories"],
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
    args = parser.parse_args()

    if args.smoketest:
        args.max_iters = 50
        args.eval_interval = 25
        args.eval_iters = 10
        args.batch_size = 16
        args.seq_len = 32

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
    models = {}
    model_info = {}

    tf = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        nlayers=5,
        dim_feedforward=256,
        max_len=args.seq_len,
    )
    models["transformer"] = tf
    model_info["transformer"] = count_params(tf)

    from velm.lite import VelmFull, VelmHybrid

    vl = VelmFull(
        vocab_size=vocab_size,
        block_size=args.block_size,
        embed_dim=128,
        latent_dim=128,
        state_dim=128,
    )
    models["velm_lite"] = vl
    model_info["velm_lite"] = count_params(vl)

    vh_fast = VelmHybrid(
        vocab_size=vocab_size,
        block_size=args.block_size,
        embed_dim=128,
        latent_dim=128,
        state_dim=88,
        num_miras_layers=2,
        num_swa_layers=1,
    )
    models["velm_hybrid_fast"] = vh_fast
    model_info["velm_hybrid_fast"] = count_params(vh_fast)

    vh_deep = VelmHybrid(
        vocab_size=vocab_size,
        block_size=args.block_size,
        embed_dim=128,
        latent_dim=64,
        state_dim=96,
        num_miras_layers=2,
        num_swa_layers=2,
    )
    models["velm_hybrid_deep"] = vh_deep
    model_info["velm_hybrid_deep"] = count_params(vh_deep)

    for name, params in model_info.items():
        print(f"  {name}: {params} params")

    # Results storage
    results = {}
    for name in models:
        results[name] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_latency": [],
            "throughput": [],
            "cib_norm": [],
        }

    # Train each model
    for name, model in models.items():
        print(f"\n--- Training {name.upper()} ---")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        t_start_step = time.time()
        for iter_num in range(args.max_iters):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
                train_loss, train_acc, cib_norm, train_lat = evaluate(
                    model,
                    train_data,
                    name,
                    args.eval_iters,
                    args.batch_size,
                    args.seq_len,
                    args.block_size,
                    device,
                )
                val_loss, val_acc, val_cib_norm, val_lat = evaluate(
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

            if name.startswith("velm"):
                out = model(X, return_cib_loss=True)
                states, latents, cib_loss = out[0], out[1], out[2]
                k_logits = out[3] if len(out) > 3 else None
                B, N, _ = states.shape
                states_flat = states.view(B * N, -1)
                # Ensure the decoder truncates block output appropriately for the exact sequence length
                logits = model.decode_block_state(states_flat)

                # Flatten the logits out and match with raw tokens, to support adaptive/remainder chunks natively
                logits = logits.view(B, N * args.block_size, -1)
                logits = logits[:, :Y.size(1), :]  # Truncate any padded elements generated by decoder
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.view(-1))
                loss = loss + 1e-3 * cib_loss
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

    # Export CSV
    history_path = os.path.join(args.out, "history.csv")
    model_order = list(results.keys())
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
        if name.startswith("velm"):
            fieldnames.append(f"{p}_cib_norm")

    with open(history_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_evals):
            row = {"step": steps[i]}
            for name in model_order:
                p = name.replace("_", "")
                n_evals = len(results[name]["train_loss"])
                if i < n_evals:
                    row[f"{p}_train_loss"] = results[name]["train_loss"][i]
                    row[f"{p}_val_loss"] = results[name]["val_loss"][i]
                    row[f"{p}_val_acc"] = results[name]["val_acc"][i]
                    row[f"{p}_val_latency"] = results[name]["val_latency"][i]
                    if name.startswith("velm") and i < len(results[name]["cib_norm"]):
                        row[f"{p}_cib_norm"] = results[name]["cib_norm"][i]
                if i > 0 and (i - 1) < len(results[name]["throughput"]):
                    row[f"{p}_throughput"] = results[name]["throughput"][i - 1]
                else:
                    row[f"{p}_throughput"] = 0.0
            writer.writerow(row)
    print(f"History saved to {history_path}")

    # Plotting
    plot_results(results, args)

    # Summary
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Vocab size: {vocab_size}\n\n")
        for name in model_order:
            f.write(f"{name.upper()} params: {model_info[name]}\n")
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
