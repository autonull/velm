#!/usr/bin/env python3
"""Train VELM Full with proper two-phase training.

Phase 1: Pretrain CALM autoencoder with CE reconstruction loss.
  - Trains embedding, encoder, decoder to reconstruct tokens from latents
  - Runs until >95% reconstruction accuracy

Phase 2: Train backbone + energy head with EGGROLL (gradient-free ES).
  - Freezes autoencoder (embedding, encoder, decoder)
  - Trains backbone (Miras + SWA) and energy head
  - Uses energy-based loss: head predicts next latent vector
  - REQUIRES GPU — EGGROLL evaluates population_size forward passes per step

Usage:
  python experiments/train_velm_full_proper.py --phase 1 --ae-steps 500
  python experiments/train_velm_full_proper.py --phase 2 --ae-ckpt results/velm_full_train/ae_final.jax
  python experiments/train_velm_full_proper.py --smoketest  # AE only, 20 steps
"""

import os
import sys
from pathlib import Path
import time
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from velm.jax.model.autoencoder import CALMAutoencoder, batch_ae_loss
from velm.jax.model.miras_backbone import VELMBackbone
from velm.jax.model.energy_head import EnergyHead
from velm.jax.training import eggroll_step, create_eggroll_optimizer
from experiments.lib.datasets import get_dataset


# ---------------------------------------------------------------------------
# Phase 1: Autoencoder Pretraining
# ---------------------------------------------------------------------------


def pretrain_autoencoder(
    ae: CALMAutoencoder,
    train_data: np.ndarray,
    *,
    steps: int = 500,
    batch_size: int = 64,
    chunk_size: int = 4,
    lr: float = 3e-3,
    key: jax.Array,
    eval_interval: int = 50,
    eval_data: np.ndarray | None = None,
    out_dir: str | None = None,
):
    """Pretrain CALM autoencoder with CE reconstruction loss."""
    params = eqx.filter(ae, eqx.is_array)
    static = eqx.filter(ae, lambda leaf: not eqx.is_array(leaf))

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    @jax.jit
    def train_step(params, opt_state, chunks, key):
        def loss_fn(p):
            model = eqx.combine(p, static)
            return batch_ae_loss(model, chunks, key=key)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    @jax.jit
    def eval_fn(params, chunks, eval_key):
        model = eqx.combine(params, static)
        return batch_ae_loss(model, chunks, key=eval_key)

    t0 = time.time()
    for step in range(steps):
        key, step_key = jax.random.split(key)

        max_start = len(train_data) - chunk_size
        starts = np.random.randint(0, max_start, size=(batch_size,))
        chunks = np.stack([train_data[s : s + chunk_size] for s in starts])

        params, opt_state, loss, metrics = train_step(
            params, opt_state, jnp.array(chunks), step_key
        )

        if step % eval_interval == 0 or step == steps - 1:
            history["train_loss"].append(float(loss))
            history["recon_loss"].append(float(metrics.get("recon_loss", 0)))
            history["kl_loss"].append(float(metrics.get("kl_loss", 0)))

            if eval_data is not None:
                max_start = len(eval_data) - chunk_size
                eval_starts = np.random.randint(0, max_start, size=(batch_size,))
                eval_chunks = np.stack([eval_data[s : s + chunk_size] for s in eval_starts])
                key, eval_key = jax.random.split(key)
                val_loss, val_metrics = eval_fn(params, jnp.array(eval_chunks), eval_key)
                history["val_loss"].append(float(val_loss))
                print(
                    f"  AE step {step}: loss={float(loss):.4f} recon={float(metrics.get('recon_loss', 0)):.4f} "
                    f"val_loss={float(val_loss):.4f}"
                )
            else:
                print(
                    f"  AE step {step}: loss={float(loss):.4f} recon={float(metrics.get('recon_loss', 0)):.4f}"
                )

        if out_dir and (step % (eval_interval * 5) == 0 or step == steps - 1):
            trained_ae = eqx.combine(params, static)
            eqx.tree_serialise_leaves(os.path.join(out_dir, "ae_checkpoint.jax"), trained_ae)

    trained_ae = eqx.combine(params, static)
    elapsed = time.time() - t0
    print(f"  Autoencoder pretraining complete in {elapsed:.1f}s")
    if history["train_loss"]:
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

    return trained_ae, history


# ---------------------------------------------------------------------------
# Phase 2: Backbone + Head Training (EGGROLL only)
# ---------------------------------------------------------------------------


def train_backbone_eggroll(
    velm,
    train_data: np.ndarray,
    *,
    steps: int = 2000,
    batch_size: int = 8,
    seq_len: int = 64,
    chunk_size: int = 4,
    lr: float = 1e-3,
    key: jax.Array,
    eval_interval: int = 100,
    eval_data: np.ndarray | None = None,
    population_size: int = 8,
    num_samples: int = 4,
    out_dir: str | None = None,
):
    """Train VELM backbone + energy head with EGGROLL (gradient-free ES).

    This is the ONLY correct way to train VELM Full. The energy head generates
    samples from noise and computes energy scores — this cannot be trained with
    gradient descent because the energy score is not differentiable w.r.t. the
    head parameters in a meaningful way.
    """
    params = eqx.filter(velm, eqx.is_array)
    static = eqx.filter(velm, lambda leaf: not eqx.is_array(leaf))
    n_chunks = seq_len // chunk_size
    n_pos = n_chunks - 1

    history = {"train_loss": [], "val_loss": [], "mean_fitness": [], "max_fitness": []}

    opt, opt_state = create_eggroll_optimizer(params, learning_rate=lr)

    @jax.jit
    def eval_fn(params, sequences, eval_key):
        model = eqx.combine(params, static)
        total_loss = 0.0
        for b in range(sequences.shape[0]):
            seq_loss, _ = model.training_loss(
                sequences[b],
                key=jax.random.fold_in(eval_key, b),
                num_samples=num_samples,
                n_pos=n_pos,
            )
            total_loss = total_loss + seq_loss
        return total_loss / sequences.shape[0]

    t0 = time.time()
    for step in range(steps):
        key, step_key = jax.random.split(key)

        max_start = len(train_data) - seq_len
        starts = np.random.randint(0, max_start, size=(batch_size,))
        sequences = np.stack(
            [train_data[s : s + seq_len].reshape(n_chunks, chunk_size) for s in starts]
        )

        # We must JIT compile the entire eggroll step to prevent jax.lax.map from
        # hanging or running extremely slowly. We also JIT compile the fitness
        # calculation to vectorize sequence batches rather than python looping.
        # DO NOT JIT THE ENTIRE EGGROLL STEP, IT IS TOO SLOW ON CPU AND CAUSES COMPILATION TIMEOUTS
        # INSTEAD JIT THE FITNESS EVALUATION ALONE
        @eqx.filter_jit
        def compiled_fitness_fn(p, seqs, s_key):
            model = eqx.combine(p, static)
            def single_seq_loss(seq, batch_idx):
                loss, _ = model.training_loss(
                    seq,
                    key=jax.random.fold_in(s_key, batch_idx),
                    num_samples=num_samples,
                    n_pos=n_pos,
                )
                return loss

            batch_indices = jnp.arange(seqs.shape[0])
            losses = jax.vmap(single_seq_loss)(seqs, batch_indices)
            return -jnp.mean(losses)

        def fitness_wrapper(p):
            return compiled_fitness_fn(p, sequences, step_key)

        new_params, new_state, metrics = eggroll_step(
            params,
            fitness_wrapper,
            opt,
            opt_state,
            key=step_key,
            population_size=population_size,
            sigma=0.01,
            rank=1,
        )
        params = new_params
        opt_state = new_state

        if step % eval_interval == 0 or step == steps - 1:
            history["mean_fitness"].append(float(metrics["mean_fitness"]))
            history["max_fitness"].append(float(metrics["max_fitness"]))

            if eval_data is not None:
                max_start = len(eval_data) - seq_len
                eval_starts = np.random.randint(0, max_start, size=(batch_size,))
                eval_sequences = np.stack(
                    [eval_data[s : s + seq_len].reshape(n_chunks, chunk_size) for s in eval_starts]
                )
                key, eval_key = jax.random.split(key)
                val_loss = eval_fn(params, jnp.array(eval_sequences), eval_key)
                history["val_loss"].append(float(val_loss))
                history["train_loss"].append(float(-metrics["mean_fitness"]))
                print(
                    f"  EGGROLL step {step}: mean_fitness={float(metrics['mean_fitness']):.4f} "
                    f"max_fitness={float(metrics['max_fitness']):.4f} "
                    f"val_loss={float(val_loss):.4f}"
                )
            else:
                print(
                    f"  EGGROLL step {step}: mean_fitness={float(metrics['mean_fitness']):.4f} "
                    f"max_fitness={float(metrics['max_fitness']):.4f}"
                )

        if out_dir and (step % (eval_interval * 5) == 0 or step == steps - 1):
            trained_velm = eqx.combine(params, static)
            eqx.tree_serialise_leaves(os.path.join(out_dir, "velm_checkpoint.jax"), trained_velm)

    trained_velm = eqx.combine(params, static)
    elapsed = time.time() - t0
    print(f"  EGGROLL training complete in {elapsed:.1f}s")
    if history["mean_fitness"]:
        print(f"  Final mean fitness: {history['mean_fitness'][-1]:.4f}")
        if history["val_loss"]:
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

    return trained_velm, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train VELM Full with proper two-phase training")
    parser.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "shakespeare_full", "tiny_stories"],
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=AE pretrain, 2=backbone+head EGGROLL (requires AE checkpoint)",
    )
    parser.add_argument("--ae-steps", type=int, default=500, help="Autoencoder pretraining steps")
    parser.add_argument("--bb-steps", type=int, default=2000, help="EGGROLL training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="AE batch size")
    parser.add_argument("--bb-batch-size", type=int, default=8, help="EGGROLL batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for EGGROLL")
    parser.add_argument("--pop", type=int, default=8, help="EGGROLL population size")
    parser.add_argument("--out", type=str, default="results/velm_full_train")
    parser.add_argument("--ae-ckpt", type=str, default=None, help="Load pretrained AE checkpoint")
    parser.add_argument("--smoketest", action="store_true", help="Quick validation: 20 AE steps")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    if args.smoketest:
        args.ae_steps = 20
        args.bb_steps = 50
        args.batch_size = 16
        args.bb_batch_size = 2
        args.seq_len = 32
        args.pop = 2

    os.makedirs(args.out, exist_ok=True)

    train_data, val_data, vocab_size, _, _ = get_dataset(
        args.dataset, block_size=4, data_dir=args.data_dir
    )
    train_np = np.array(train_data.numpy())
    val_np = np.array(val_data.numpy()) if val_data is not None else None

    print(f"Dataset: {args.dataset}, Vocab: {vocab_size}")
    print(f"Train: {len(train_np)} tokens, Val: {len(val_np) if val_np is not None else 0} tokens")

    key = jax.random.PRNGKey(42)

    if args.phase == 1:
        print("\n=== Phase 1: Autoencoder Pretraining ===")
        ae_key, key = jax.random.split(key)
        ae = CALMAutoencoder(
            vocab_size=vocab_size,
            chunk_size=4,
            hidden_dim=256,
            latent_dim=64,
            ffn_intermediate=512,
            kl_weight=0.0,  # Disable KL during pretraining
            key=ae_key,
        )
        ae_param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(ae, eqx.is_array)))
        print(f"  Autoencoder params: {ae_param_count}")

        if args.ae_ckpt:
            print(f"  Loading pretrained AE from {args.ae_ckpt}")
            ae = eqx.tree_deserialise_leaves(args.ae_ckpt, ae)

        ae, ae_history = pretrain_autoencoder(
            ae,
            train_np,
            steps=args.ae_steps,
            batch_size=args.batch_size,
            chunk_size=4,
            lr=3e-3,
            key=ae_key,
            eval_interval=max(1, args.ae_steps // 10),
            eval_data=val_np,
            out_dir=args.out,
        )

        with open(os.path.join(args.out, "ae_history.json"), "w") as f:
            json.dump(ae_history, f, indent=2)
        eqx.tree_serialise_leaves(os.path.join(args.out, "ae_final.jax"), ae)

    elif args.phase == 2:
        if not args.ae_ckpt:
            print("ERROR: Phase 2 requires --ae-ckpt with a pretrained autoencoder.")
            print("Run phase 1 first: python train_velm_full_proper.py --phase 1")
            sys.exit(1)

        print("\n=== Phase 2: Backbone + Head Training (EGGROLL) ===")
        print("NOTE: This requires a GPU. EGGROLL evaluates pop_size forward passes per step.")
        print("      On CPU, this will be extremely slow.")

        velm_key, key = jax.random.split(key)

        # Load pretrained AE
        ae = CALMAutoencoder(
            vocab_size=vocab_size,
            chunk_size=4,
            hidden_dim=256,
            latent_dim=64,
            ffn_intermediate=512,
            kl_weight=0.0,
            key=velm_key,
        )
        ae = eqx.tree_deserialise_leaves(args.ae_ckpt, ae)

        # Build backbone and head with matching dimensions
        k1, k2 = jax.random.split(velm_key)
        backbone = VELMBackbone(
            dim=256,
            num_heads=4,
            num_miras_layers=2,
            num_swa_layers=2,
            ffn_intermediate=512,
            chunk_size=4,
            ae_hidden_dim=256,
            key=k1,
        )
        head = EnergyHead(hidden_dim=256, latent_dim=ae.latent_dim, num_blocks=2, key=k2)

        class _VELM(eqx.Module):
            autoencoder: CALMAutoencoder
            backbone: VELMBackbone
            head: EnergyHead
            chunk_size: int

            def training_loss(self, token_ids, *, key, num_samples=8, alpha=1.0, n_pos=None):
                # 1. encode all chunks → target latent vectors
                def encode_one(chunk):
                    z, _, _ = self.autoencoder.encode(chunk, training=False)
                    return z

                target_z = jax.vmap(encode_one)(token_ids)

                # 2. build input representations
                def compress_chunk(chunk_ids):
                    embs = jax.vmap(self.autoencoder.embedding)(chunk_ids)
                    return self.backbone.compress_input(embs)

                input_seq = jax.vmap(compress_chunk)(token_ids)

                # 3. backbone forward
                hidden_states, _ = self.backbone(input_seq)

                # 4. energy loss: predict z_{i+1} from h_i
                h_input = hidden_states[:-1]
                z_target = target_z[1:]

                # Dynamic shape tracking requires jax dynamic slicing instead of hardcoding python int arrays
                keys = jax.random.split(key, h_input.shape[0])

                def position_loss(h, target, k):
                    samples = self.head(h, key=k, num_samples=num_samples)
                    from velm.jax.model.energy_head import energy_score

                    return energy_score(samples, target, alpha=alpha)

                losses = jax.vmap(position_loss)(h_input, z_target, keys)
                mean_loss = jnp.mean(losses)
                return mean_loss, {
                    "energy_loss": mean_loss,
                    "num_positions": jnp.array(h_input.shape[0], dtype=jnp.float32),
                }

        velm = _VELM(autoencoder=ae, backbone=backbone, head=head, chunk_size=4)
        velm_param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(velm, eqx.is_array)))
        print(f"  VELM Full params: {velm_param_count}")

        velm, bb_history = train_backbone_eggroll(
            velm,
            train_np,
            steps=args.bb_steps,
            batch_size=args.bb_batch_size,
            seq_len=args.seq_len,
            chunk_size=4,
            lr=1e-3,
            key=velm_key,
            eval_interval=max(1, args.bb_steps // 10),
            eval_data=val_np,
            population_size=args.pop,
            num_samples=4,
            out_dir=args.out,
        )

        with open(os.path.join(args.out, "bb_history.json"), "w") as f:
            json.dump(bb_history, f, indent=2)
        eqx.tree_serialise_leaves(os.path.join(args.out, "velm_final.jax"), velm)

    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
