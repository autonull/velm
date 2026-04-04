#!/usr/bin/env python3
"""VELM vs Transformer benchmark — consolidated experiment.

Replaces the three previous mini_velm_experiment* scripts with a single
entry point that supports multiple modes via flags:

  --mode basic       Train both models for fixed steps (fast, default)
  --mode tuned       Quick hyperparameter sweep + param-matched Transformer
  --mode extended    Tuned + qTTT adaptation sweep + time-limited training

All outputs go to results/<experiment_name>/ by default.
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch

# Ensure experiments/ is on the path so `lib` is importable when running standalone
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from lib.data import generate_delayed_recall
from lib.models import (
    VelmProxy,
    TransformerLM,
    count_params,
    find_transformer_config,
    evaluate_model,
    train_model,
    train_model_time_limited,
)
from lib.plots import save_plots

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

QID = 1


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def collect_activations(velm_model, tf_model, test_x, batch_size, device):
    """Collect sample activation matrices for visualization."""
    sample = test_x[:batch_size].to(device)
    hist = sample[:, :-1]
    with torch.no_grad():
        tf_logits, _ = tf_model(hist, return_attn=True)
        velm_states, _ = velm_model(hist)
    tf_acts = tf_logits[0].cpu().numpy()
    tf_acts_small = tf_acts[:, : min(64, tf_acts.shape[1])]
    velm_acts = velm_states[0].cpu().numpy()
    return {"transformer": tf_acts_small, "velm": velm_acts}


# ---------------------------------------------------------------------------
# Mode: basic — fixed-step training
# ---------------------------------------------------------------------------


def run_basic(args, device):
    VOCAB = 64
    SEQ_LEN = 65
    BLOCK = 4

    train_x, train_gaps, _ = generate_delayed_recall(args.train_n, SEQ_LEN, VOCAB, max_gap=20)
    test_x, test_gaps, _ = generate_delayed_recall(args.test_n, SEQ_LEN, VOCAB, max_gap=20)

    tf_model = TransformerLM(
        VOCAB, d_model=64, nhead=4, nlayers=2, dim_feedforward=128, max_len=SEQ_LEN - 1
    ).to(device)
    velm_model = VelmProxy(VOCAB, embed_dim=64, latent_dim=48, memory_dim=128, block_size=BLOCK).to(
        device
    )

    print("Training Transformer...")
    tf_losses = train_model(
        tf_model,
        train_x,
        train_gaps,
        steps=args.steps,
        batch_size=args.batch,
        model_name="transformer",
        block_size=BLOCK,
        device=device,
    )
    print("Training VELM...")
    velm_losses = train_model(
        velm_model,
        train_x,
        train_gaps,
        steps=args.steps,
        batch_size=args.batch,
        cib_lambda=1e-3,
        model_name="velm",
        block_size=BLOCK,
        device=device,
    )

    tf_acc, tf_gap = evaluate_model(
        tf_model, test_x, test_gaps, args.batch, "transformer", block_size=BLOCK, device=device
    )
    velm_acc, velm_gap = evaluate_model(
        velm_model, test_x, test_gaps, args.batch, "velm", block_size=BLOCK, device=device
    )

    print(f"\nTransformer: acc={tf_acc:.4f}  params={count_params(tf_model)}")
    print(f"VELM:        acc={velm_acc:.4f}  params={count_params(velm_model)}")

    acts = collect_activations(velm_model, tf_model, test_x, args.batch, device)
    save_plots(
        args.out,
        {"transformer": tf_losses, "velm": velm_losses},
        {"transformer": tf_acc, "velm": velm_acc},
        {"transformer": tf_gap, "velm": velm_gap},
        acts,
    )

    with open(os.path.join(args.out, "summary.txt"), "a") as f:
        f.write(f"Transformer params={count_params(tf_model)} acc={tf_acc:.4f}\n")
        f.write(f"VELM params={count_params(velm_model)} acc={velm_acc:.4f}\n")

    print(f"Results saved to {args.out}/")
    return {"velm_acc": velm_acc, "tf_acc": tf_acc}


# ---------------------------------------------------------------------------
# Mode: tuned — hyperparameter sweep + param-matched Transformer
# ---------------------------------------------------------------------------


def run_tuned(args, device):
    VOCAB = 64
    SEQ_LEN = 65
    BLOCK = 4
    L = SEQ_LEN - 1

    train_x, train_gaps, _ = generate_delayed_recall(args.train_n, SEQ_LEN, VOCAB, max_gap=20)
    test_x, test_gaps, _ = generate_delayed_recall(args.test_n, SEQ_LEN, VOCAB, max_gap=20)

    candidates = [
        {"latent_dim": 48, "memory_dim": 128, "cib_lambda": 1e-3, "lr": 1e-3},
        {"latent_dim": 64, "memory_dim": 192, "cib_lambda": 5e-4, "lr": 5e-4},
        {"latent_dim": 96, "memory_dim": 256, "cib_lambda": 1e-4, "lr": 5e-4},
    ]

    tune_each = max(5, args.tune_time // len(candidates))
    best_cfg, best_acc = None, -1.0

    print("Hyperparameter tuning...")
    for cfg in candidates:
        m = VelmProxy(
            VOCAB,
            embed_dim=64,
            latent_dim=cfg["latent_dim"],
            memory_dim=cfg["memory_dim"],
            block_size=BLOCK,
        )
        _, t = train_model_time_limited(
            m,
            train_x,
            train_gaps,
            time_budget_sec=tune_each,
            batch_size=args.batch,
            lr=cfg["lr"],
            cib_lambda=cfg["cib_lambda"],
            model_name="velm",
            block_size=BLOCK,
            device=device,
            max_steps=10000,
        )
        acc, _ = evaluate_model(
            m, test_x, test_gaps, args.batch, "velm", block_size=BLOCK, device=device
        )
        print(f"  cfg={cfg}  acc={acc:.4f}  time={t:.1f}s")
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg

    print(f"Best: {best_cfg}  acc={best_acc:.4f}")

    # Final VELM
    velm_final = VelmProxy(
        VOCAB,
        embed_dim=64,
        latent_dim=best_cfg["latent_dim"],
        memory_dim=best_cfg["memory_dim"],
        block_size=BLOCK,
    )
    velm_losses, velm_time = train_model_time_limited(
        velm_final,
        train_x,
        train_gaps,
        time_budget_sec=args.final_time,
        batch_size=args.batch,
        lr=best_cfg["lr"],
        cib_lambda=best_cfg["cib_lambda"],
        model_name="velm",
        block_size=BLOCK,
        device=device,
        max_steps=1000000,
    )
    velm_acc, velm_gap = evaluate_model(
        velm_final, test_x, test_gaps, args.batch, "velm", block_size=BLOCK, device=device
    )
    velm_params = count_params(velm_final)
    print(f"VELM: params={velm_params} acc={velm_acc:.4f} time={velm_time:.1f}s")

    # Param-matched Transformer
    d_model, nhead, nlayers, tf_params = find_transformer_config(VOCAB, L, velm_params)
    print(f"Transformer matched: d={d_model} heads={nhead} layers={nlayers} params={tf_params}")
    tf_model = TransformerLM(
        VOCAB,
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dim_feedforward=max(32, d_model * 2),
        max_len=L,
    )
    tf_time_budget = max(5, int(args.final_time * 0.6))
    tf_losses, tf_time = train_model_time_limited(
        tf_model,
        train_x,
        train_gaps,
        time_budget_sec=tf_time_budget,
        batch_size=args.batch,
        lr=1e-3,
        model_name="transformer",
        block_size=BLOCK,
        device=device,
        max_steps=1000000,
    )
    tf_acc, tf_gap = evaluate_model(
        tf_model, test_x, test_gaps, args.batch, "transformer", block_size=BLOCK, device=device
    )
    print(f"Transformer: params={tf_params} acc={tf_acc:.4f} time={tf_time:.1f}s")

    acts = collect_activations(velm_final, tf_model, test_x, args.batch, device)
    save_plots(
        args.out,
        {"transformer": tf_losses, "velm": velm_losses},
        {"transformer": tf_acc, "velm": velm_acc},
        {"transformer": tf_gap, "velm": velm_gap},
        acts,
    )

    with open(os.path.join(args.out, "summary.txt"), "a") as f:
        f.write(f"velm_params={velm_params}\nvelm_acc={velm_acc:.4f}\nvelm_time={velm_time:.1f}\n")
        f.write(
            f"transformer_params={tf_params}\ntransformer_acc={tf_acc:.4f}\ntransformer_time={tf_time:.1f}\n"
        )
        f.write(f"best_cfg={best_cfg}\n")

    print(f"Results saved to {args.out}/")
    return {
        "velm_acc": velm_acc,
        "tf_acc": tf_acc,
        "velm_params": velm_params,
        "tf_params": tf_params,
    }


# ---------------------------------------------------------------------------
# Mode: extended — tuned + qTTT sweep
# ---------------------------------------------------------------------------


def run_extended(args, device):
    """Extended mode: tuned comparison + qTTT adaptation sweep."""
    results = run_tuned(args, device)

    # qTTT adaptation sweep on the trained VELM model
    # (For the proxy, this means testing adapter-based adaptation)
    print("\n--- qTTT Adaptation Sweep ---")
    from velm.torch_proxy import VelmFull

    VOCAB = 128
    SEQ_LEN = 33
    model = VelmFull(vocab_size=VOCAB, block_size=4, embed_dim=32, latent_dim=48, state_dim=96).to(
        device
    )

    train_x, train_gaps, _ = generate_delayed_recall(400, SEQ_LEN, VOCAB, max_gap=12)
    test_x, test_gaps, _ = generate_delayed_recall(100, SEQ_LEN, VOCAB, max_gap=12)

    # Pre-adaptation accuracy
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        idx = torch.randint(0, train_x.shape[0], (64,))
        batch = train_x[idx].to(device)
        hist = batch[:, :-1]
        qpos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), qpos + 1].to(device)
        states, _ = model(hist)
        b_idx = (qpos // model.block_size).long()
        logits = model.decode_block_state(states[torch.arange(batch.shape[0]), b_idx, :])
        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pre_acc, _ = evaluate_model(model, test_x, test_gaps, 64, "velm", block_size=4, device=device)

    # Adaptation: fine-tune adapter only
    adapter_params = [p for n, p in model.named_parameters() if "adapter" in n]
    if adapter_params:
        opt_adapter = torch.optim.Adam(adapter_params, lr=5e-3)
        for step in range(25):
            idx = torch.randint(0, train_x.shape[0], (64,))
            batch = train_x[idx].to(device)
            hist = batch[:, :-1]
            qpos = (hist == QID).int().argmax(dim=1)
            targets = batch[torch.arange(batch.shape[0]), qpos + 1].to(device)
            states, _ = model(hist, use_adapter=True)
            b_idx = (qpos // model.block_size).long()
            logits = model.decode_block_state(states[torch.arange(batch.shape[0]), b_idx, :])
            loss = torch.nn.functional.cross_entropy(logits[:, -1, :], targets)
            opt_adapter.zero_grad()
            loss.backward()
            opt_adapter.step()

    post_acc, _ = evaluate_model(model, test_x, test_gaps, 64, "velm", block_size=4, device=device)

    print(f"qTTT pre-adapter acc: {pre_acc:.4f}")
    print(f"qTTT post-adapter acc: {post_acc:.4f}")
    print(f"qTTT delta: {post_acc - pre_acc:+.4f}")

    with open(os.path.join(args.out, "summary.txt"), "a") as f:
        f.write(
            f"qttt_pre_acc={pre_acc:.4f}\nqttt_post_acc={post_acc:.4f}\nqttt_delta={post_acc - pre_acc:+.4f}\n"
        )

    # qTTT adaptation plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(["pre", "post"], [pre_acc, post_acc], color=["#440154", "#fde725"])
    plt.ylim(0, 1)
    plt.title("qTTT Adapter Adaptation")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "qttt_adapt.png"))
    plt.close()

    results["qttt_pre"] = pre_acc
    results["qttt_post"] = post_acc
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="VELM vs Transformer benchmark")
    parser.add_argument(
        "--mode",
        choices=["basic", "tuned", "extended"],
        default="basic",
        help="Experiment mode: basic (fast), tuned (param-matched), extended (+qTTT)",
    )
    parser.add_argument("--steps", type=int, default=300, help="Training steps (basic mode)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out", type=str, default=None, help="Output directory (default: results/<mode>)"
    )
    parser.add_argument("--train_n", type=int, default=2000)
    parser.add_argument("--test_n", type=int, default=500)
    parser.add_argument(
        "--tune_time", type=int, default=30, help="Seconds for hyperparam tuning (tuned/extended)"
    )
    parser.add_argument(
        "--final_time", type=int, default=120, help="Seconds for final training (tuned/extended)"
    )
    parser.add_argument(
        "--smoketest", action="store_true", help="Quick smoke test: 10 steps, tiny data"
    )
    args = parser.parse_args()

    if args.smoketest:
        args.steps = 10
        args.train_n = 64
        args.test_n = 16
        args.tune_time = 5
        args.final_time = 10
        args.batch = 16

    device = resolve_device(args.device)
    args.out = args.out or f"results/{args.mode}"

    print(f"VELM vs Transformer — mode={args.mode}  device={device}  out={args.out}")
    print(f"{'[SMOKETEST] ' if args.smoketest else ''}Starting experiment...\n")

    t0 = time.time()
    if args.mode == "basic":
        results = run_basic(args, device)
    elif args.mode == "tuned":
        results = run_tuned(args, device)
    elif args.mode == "extended":
        results = run_extended(args, device)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {args.out}/")
    return results


if __name__ == "__main__":
    main()
