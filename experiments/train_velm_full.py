#!/usr/bin/env python3
"""Train and smoke-test the VelmFull model on a tiny synthetic task.

Produces outputs in results/train_velm_full/ by default. GPU-ready; will use CUDA if available.
Supports optional EGGROLL evolutionary strategy for adapter tuning.
"""

import os
import sys
from pathlib import Path

# Ensure src/ is on the path so `velm.*` imports work when running standalone
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from velm.lite import VelmFull
from velm.lite.cib import CIBLoss
from velm.lite.eggroll_stub import Eggroll

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

QID = 1


def generate_dataset(n, seq_len, vocab, max_gap=12):
    L = seq_len - 1
    # use torch RNG to avoid numpy version issues
    gaps = torch.randint(1, max_gap + 1, (n,), dtype=torch.long).numpy()
    data = torch.randint(3, vocab, (n, seq_len), dtype=torch.long).numpy()
    for i in range(n):
        g = int(gaps[i])
        lbl = int(torch.randint(3, vocab, (1,)).item())
        data[i, 0] = lbl
        if g >= seq_len - 1:
            g = seq_len - 2
        data[i, g] = QID
        data[i, g + 1] = lbl
    return torch.tensor(data, dtype=torch.long), torch.tensor(gaps, dtype=torch.long)


def eval_accuracy(model, data, gaps, device, n_eval=128):
    model.to(device)
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, data.shape[0], (n_eval,))
        batch = data[idx.long()].to(device)
        hist = batch[:, :-1]
        qpos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), qpos + 1].to(device)
        states, latents = model(hist, use_adapter=True)
        b_idx = (qpos // model.block_size).long()
        state_at = states[torch.arange(batch.shape[0]), b_idx, :]
        logits = model.decode_block_state(state_at)
        # Model now decodes K tokens. We just want the single target token corresponding to the question.
        # Since it's a simplification, we just predict the last token of the block to align with original proxy.
        logits_last = logits[:, -1, :]
        preds = logits_last.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
    model.train()
    return acc


def train_one_epoch(model, data, gaps, device, steps=50, batch_size=64, cib_lambda=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cib = CIBLoss(lambda_c=cib_lambda)
    N = data.shape[0]
    losses = []
    for s in range(steps):
        idx = torch.randint(N, (batch_size,))
        batch_data = data[idx.long()].to(device)
        hist = batch_data[:, :-1]
        qpos = (hist == QID).int().argmax(dim=1)
        targets = batch_data[torch.arange(batch_data.shape[0]), qpos + 1].to(device)
        states, latents = model(hist, use_adapter=False)
        # pick the state corresponding to query block
        b_idx = (qpos // model.block_size).long()
        state_at = states[torch.arange(batch_data.shape[0]), b_idx, :]
        logits = model.decode_block_state(state_at)
        # Model now decodes K tokens. We just want the single target token corresponding to the question.
        logits_last = logits[:, -1, :]
        loss = F.cross_entropy(logits_last, targets)
        loss = cib(loss, latents[torch.arange(batch_data.shape[0]), b_idx, :])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/train_velm_full")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eggroll", action="store_true", help="Run EGGROLL ES tuning after Adam training"
    )
    parser.add_argument("--pop", type=int, default=8, help="EGGROLL population size")
    parser.add_argument("--gens", type=int, default=3, help="EGGROLL generations")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker threads for ES evaluation"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Use multiprocessing pool for ES evaluation (requires picklable fitness_fn)",
    )
    parser.add_argument(
        "--evolve", choices=["adapter", "all"], default="adapter", help="Which params to evolve"
    )
    parser.add_argument("--alg", choices=["es", "cma"], default="es", help="ES algorithm")
    parser.add_argument(
        "--sweep", action="store_true", help="Run a hyperparameter sweep over EGGROLL parameters"
    )
    parser.add_argument(
        "--smoketest", action="store_true", help="Quick smoke test: 5 steps, pop=2, gens=1"
    )
    args = parser.parse_args()

    if args.smoketest:
        args.steps = 5
        args.pop = 2
        args.gens = 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print("Device:", device)
    vocab = 128
    seq_len = 33
    train_x, train_gaps = generate_dataset(800, seq_len, vocab, max_gap=12)
    model = VelmFull(vocab_size=vocab, block_size=4, embed_dim=32, latent_dim=48, state_dim=96)
    t0 = time.time()
    losses = train_one_epoch(
        model, train_x, train_gaps, device, steps=args.steps, batch_size=128, cib_lambda=1e-3
    )
    t = time.time() - t0
    print("Train time:", t)

    # compute pre-ES accuracy and save losses
    acc_pre = eval_accuracy(model, train_x, train_gaps, device, n_eval=256)
    try:
        import matplotlib.pyplot as plt

        has_plt = True
    except Exception:
        has_plt = False

    if has_plt:
        os.makedirs(args.out, exist_ok=True)
        plt.figure()
        plt.plot(losses)
        plt.title("Train loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.savefig(os.path.join(args.out, "loss.png"))
        plt.close()

    with open(os.path.join(args.out, "prepost.txt"), "w") as f:
        f.write(f"pre_acc={acc_pre}\n")

    if args.sweep:
        print("Running EGGROLL Hyperparameter Sweep...")
        import csv
        import copy

        # Define hyperparameter grid
        algs = ["es", "cma"]
        sigmas = [1e-2, 2e-2, 5e-2]
        lrs = [0.05, 0.1, 0.2]

        sweep_results = []
        best_improvement = -float("inf")
        best_params = None

        # To ensure fair comparison, evaluate starting from the same base model
        base_model_state = copy.deepcopy(model.state_dict())

        for alg in algs:
            for sigma in sigmas:
                for lr in lrs:
                    print(f"Testing config: alg={alg}, sigma={sigma}, lr={lr}")
                    # reset model to pre-ES state
                    model.load_state_dict(copy.deepcopy(base_model_state))

                    es = Eggroll(sigma=sigma)

                    def fitness(m):
                        return eval_accuracy(m, train_x, train_gaps, device, n_eval=128)

                    t_sweep_start = time.time()
                    try:
                        success = es.run_es(
                            model,
                            fitness,
                            pop_size=args.pop,
                            generations=args.gens,
                            sigma=sigma,
                            lr=lr,
                            device=device,
                            num_workers=args.workers,
                            evolve=args.evolve,
                            algorithm=alg,
                            use_processes=args.process,
                        )
                        acc_post = eval_accuracy(model, train_x, train_gaps, device, n_eval=256)
                    except Exception as e:
                        print(f"Run failed for {alg}, sigma={sigma}, lr={lr}: {e}")
                        success = False
                        acc_post = 0.0
                    t_sweep_end = time.time()

                    improvement = acc_post - acc_pre

                    sweep_results.append(
                        {
                            "alg": alg,
                            "sigma": sigma,
                            "lr": lr,
                            "pre_acc": acc_pre,
                            "post_acc": acc_post,
                            "improvement": improvement,
                            "time_s": t_sweep_end - t_sweep_start,
                        }
                    )

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_params = {"alg": alg, "sigma": sigma, "lr": lr, "post_acc": acc_post}

        print("\n--- Sweep Completed ---")
        print(f"Best Improvement: {best_improvement:.4f} with params: {best_params}")

        # Write results to CSV
        csv_path = os.path.join(args.out, "sweep_results.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["alg", "sigma", "lr", "pre_acc", "post_acc", "improvement", "time_s"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in sweep_results:
                writer.writerow(row)
        print(f"Sweep results saved to {csv_path}")

    elif args.eggroll:
        print("Running EGGROLL ES tuning (Multi-Island GEA with Diversity Bonus)...")
        import copy

        num_islands = max(1, args.pop // 4)
        islands = [copy.deepcopy(model) for _ in range(num_islands)]
        es_instances = [Eggroll(sigma=2e-2) for _ in range(num_islands)]

        # We need a shared reference pool for computing diversity bonus
        # Store latest evaluated state dict vectors.
        # This will be updated by fitness functions.
        shared_vectors = []

        def get_model_vector(m):
            vecs = []
            for p in m.parameters():
                if p.requires_grad:
                    vecs.append(p.detach().reshape(-1))
            return torch.cat(vecs)

        def create_fitness_fn(island_idx):
            def fitness(m):
                # 1. Base Fitness (Accuracy)
                acc = eval_accuracy(m, train_x, train_gaps, device, n_eval=128)

                # 2. Generalization Bonus (Synthetic Reasoning)
                # For this proxy, we evaluate on a slightly larger gap to encourage robust reasoning
                train_x_hard, train_gaps_hard = generate_dataset(128, seq_len, vocab, max_gap=16)
                acc_gen = eval_accuracy(m, train_x_hard, train_gaps_hard, device, n_eval=128)

                # 3. CIB Compression / Structural Penalty
                # Lower norm is better, but accuracy should be high.
                _, latents = m(train_x[:32].to(device))
                cib_norm = latents.norm(p=2, dim=2).mean().item()
                compression_bonus = 1.0 / (1.0 + cib_norm)

                # 4. Diversity Bonus (Pairwise Cosine distance)
                # Compare this model's parameters to the running shared history pool
                div_bonus = 0.0
                if len(shared_vectors) > 0:
                    vec = get_model_vector(m).cpu()
                    # Sample up to 5 random vectors from history
                    sample_size = min(5, len(shared_vectors))
                    indices = np.random.choice(len(shared_vectors), sample_size, replace=False)
                    dists = []
                    for idx in indices:
                        other_vec = shared_vectors[idx]
                        if vec.shape == other_vec.shape:
                            cos_sim = F.cosine_similarity(vec.unsqueeze(0), other_vec.unsqueeze(0)).item()
                            # 1 - cos_sim -> 0 if same, 2 if opposite.
                            dists.append(1.0 - cos_sim)
                    if dists:
                        div_bonus = np.mean(dists)

                # Add this model's vector to the shared pool (with simple reservoir sampling to prevent memory blowout)
                if len(shared_vectors) < 100:
                    shared_vectors.append(get_model_vector(m).cpu())
                elif np.random.rand() < 0.1:
                    shared_vectors[np.random.randint(0, 100)] = get_model_vector(m).cpu()

                # GEA 2.0 fitness
                return (acc * 0.5 + acc_gen * 0.5) * compression_bonus + 0.1 * div_bonus
            return fitness

        # Multi-Island Evolution Loop
        generations_per_migration = max(1, args.gens // 3)
        num_migrations = max(1, args.gens // generations_per_migration)

        for mig_idx in range(num_migrations):
            print(f"Migration epoch {mig_idx+1}/{num_migrations}")
            for island_idx in range(num_islands):
                es_instances[island_idx].run_es(
                    islands[island_idx],
                    create_fitness_fn(island_idx),
                    pop_size=max(2, args.pop // num_islands),
                    generations=generations_per_migration,
                    sigma=2e-2,
                    lr=0.1,
                    device=device,
                    num_workers=args.workers,
                    evolve=args.evolve,
                    algorithm=args.alg,
                    use_processes=args.process,
                )

            # Experience Migration:
            # Find the best performing island and share its weights with the worst
            island_fitnesses = []
            for i in range(num_islands):
                f_fn = create_fitness_fn(i)
                island_fitnesses.append(f_fn(islands[i]))

            best_island_idx = np.argmax(island_fitnesses)
            worst_island_idx = np.argmin(island_fitnesses)

            if best_island_idx != worst_island_idx:
                print(f"  Migrating experience from Island {best_island_idx} to Island {worst_island_idx}")
                islands[worst_island_idx].load_state_dict(islands[best_island_idx].state_dict())

        # Select the globally best model after all migrations
        final_fitnesses = [create_fitness_fn(i)(islands[i]) for i in range(num_islands)]
        best_island = islands[np.argmax(final_fitnesses)]
        model.load_state_dict(best_island.state_dict())

        print("EGGROLL completed (Multi-Island GEA)")
        acc_post = eval_accuracy(model, train_x, train_gaps, device, n_eval=256)
        with open(os.path.join(args.out, "eggroll_result.txt"), "w") as f:
            f.write(f"eggroll_post_acc={acc_post}\n")
        # save pre/post plot if matplotlib available
        if has_plt:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.bar([0, 1], [acc_pre, acc_post], tick_label=["pre", "post"])
            plt.ylim(0, 1)
            plt.title("Pre/Post EGGROLL accuracy")
            plt.savefig(os.path.join(args.out, "prepost_acc.png"))
            plt.close()

    # save a tiny artifact to show success
    with open(os.path.join(args.out, "smoke.txt"), "w") as f:
        f.write(f"trained_steps={len(losses)} time={t} pre_acc={acc_pre}\n")


if __name__ == "__main__":
    main()
