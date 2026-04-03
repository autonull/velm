#!/usr/bin/env python3
"""
Mini VELM vs Transformer experiment (fast proxy)

Generates a synthetic delayed-recall dataset and trains two tiny models:
- VELM proxy: CALM-like block compression + simple MLP memory + CIB penalty
- Transformer baseline: small causal transformer

Saves plots and metrics to outputs/ and prints a short summary.
Designed to run in minutes on a single 8-16GB GPU or CPU.
"""

import os
import time
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data generation ---

def generate_dataset(n_samples, seq_len, vocab_size, max_gap=None):
    # seq_len is T (we will use history length L = T-1 for inputs)
    # tokens: 0..vocab_size-1
    # special tokens: QID=1 (query); we reserve ids 0..2 (0 unused, 1=Q)
    QID = 1
    assert seq_len >= 8
    L = seq_len - 1
    if max_gap is None:
        max_gap = max(2, L // 2)
    gaps = np.random.randint(1, max_gap + 1, size=n_samples)
    data = np.random.randint(3, vocab_size, size=(n_samples, seq_len))
    labels = np.random.randint(3, vocab_size, size=n_samples)
    for i in range(n_samples):
        g = int(gaps[i])
        Lbl = int(labels[i])
        # put label at position 0 (memory token)
        data[i, 0] = Lbl
        # put query marker at position g (1 <= g <= L-1)
        # ensure g+1 < seq_len
        if g >= seq_len - 1:
            g = seq_len - 2
            gaps[i] = g
        data[i, g] = QID
        # set the next token after query to be the label (the model must predict this)
        data[i, g + 1] = Lbl
    return torch.tensor(data, dtype=torch.long), torch.tensor(gaps, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# --- Models ---

class VelmProxy(nn.Module):
    """Tiny VELM proxy: block-based CALM compression + simple MLP memory + linear decoder"""
    def __init__(self, vocab_size, embed_dim=64, latent_dim=48, memory_dim=128, block_size=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.block_size = block_size
        self.block_proj = nn.Linear(embed_dim, latent_dim)
        # memory MLP that maps a block-latent to a delta state
        self.mem_mlp = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.decoder = nn.Linear(memory_dim, vocab_size)

    def forward(self, inp):
        # inp: (B, L) history tokens (we compress into blocks of size block_size)
        B, L = inp.shape
        assert L % self.block_size == 0, "History length must be divisible by block_size"
        n_blocks = L // self.block_size
        emb = self.embed(inp)  # (B, L, E)
        emb_blocks = emb.view(B, n_blocks, self.block_size, -1)
        block_mean = emb_blocks.mean(dim=2)  # (B, n_blocks, E)
        latents = self.block_proj(block_mean)  # (B, n_blocks, latent)
        # run sequential memory and collect per-block states
        memory_dim = self.mem_mlp[-1].out_features
        state = torch.zeros(B, memory_dim, device=latents.device)
        states = []
        for i in range(n_blocks):
            delta = self.mem_mlp(latents[:, i, :])
            state = state + delta  # simple associative update
            states.append(state.unsqueeze(1))
        states = torch.cat(states, dim=1)  # (B, n_blocks, memory_dim)
        return states, latents


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x: (B, L, d_model)
        attn_out, attn_w = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x, attn_w


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, nlayers=2, dim_feedforward=128, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(nlayers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, inp, return_attn=False):
        # inp: (B, L)
        B, L = inp.shape
        x = self.embed(inp) + self.pos[:L].unsqueeze(0).to(inp.device)
        # causal mask
        mask = torch.triu(torch.full((L, L), float('-inf'), device=inp.device), diagonal=1)
        attns = []
        for layer in self.layers:
            x, a = layer(x, attn_mask=mask)
            attns.append(a)
        logits = self.out(x)  # (B, L, V)
        if return_attn:
            return logits, attns
        return logits

# --- Utilities ---

def batch_indices(n, batch_size):
    idx = np.random.randint(0, n, size=batch_size)
    return idx


def evaluate_model(model, dataset_inputs, dataset_gaps, batch_size, model_name, block_size=4):
    model.eval()
    inputs = dataset_inputs
    gaps = dataset_gaps
    N = inputs.shape[0]
    correct = 0
    per_gap = defaultdict(lambda: [0, 0])
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = inputs[i: i + batch_size].to(device)
            batch_gaps = gaps[i: i + batch_size].to(device)
            # history inputs are all tokens except last one
            hist = batch[:, :-1]
            if model_name == 'transformer':
                logits = model(hist)  # (B, L, V)
                # gather logits at positions q_pos (g), targets at g+1
                q_pos = (hist == 1).int().argmax(dim=1)  # (B,)
                logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            else:
                # VELM: compute per-block states
                states, latents = model(hist)
                q_pos = (hist == 1).int().argmax(dim=1)
                b_idx = (q_pos // block_size).long()
                logits_at = model.decoder(states[torch.arange(states.shape[0]), b_idx, :])
            targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
            pred = logits_at.argmax(dim=1)
            correct += (pred == targets).sum().item()
            for j in range(batch.shape[0]):
                g = int(batch_gaps[j].item())
                ok = int((pred[j] == targets[j]).item())
                per_gap[g][0] += ok
                per_gap[g][1] += 1
    total = N
    acc = correct / total
    # compute per-gap accuracy (sorted)
    gaps_sorted = sorted(per_gap.keys())
    gap_acc = [(g, per_gap[g][0] / per_gap[g][1]) for g in gaps_sorted]
    return acc, gap_acc


# --- Training procedure ---

def train_model(model, train_inputs, train_gaps, steps=300, batch_size=128, lr=1e-3, cib_lambda=1e-3, model_name='transformer', block_size=4):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = train_inputs.shape[0]
    for step in range(1, steps + 1):
        idx = batch_indices(N, batch_size)
        batch = train_inputs[idx].to(DEVICE)
        batch_gaps = train_gaps[idx].to(DEVICE)
        hist = batch[:, :-1]
        # find Q positions
        q_pos = (hist == 1).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(DEVICE)
        if model_name == 'transformer':
            logits = model(hist)  # (B, L, V)
            logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            loss = F.cross_entropy(logits_at, targets)
        else:
            states, latents = model(hist)
            b_idx = (q_pos // block_size).long()
            logits_at = model.decoder(states[torch.arange(states.shape[0]), b_idx, :])
            loss = F.cross_entropy(logits_at, targets)
            # CIB proxy: L2 penalty on latents averaged over blocks used (query block latents)
            lat_q = latents[torch.arange(latents.shape[0]), b_idx, :]
            cib = lat_q.norm(p=2, dim=1).mean()
            loss = loss + cib_lambda * cib

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, steps // 10) == 0 or step <= 5:
            print(f"[{model_name}] step {step}/{steps} loss={loss.item():.4f}")
    return losses


def plot_results(out_dir, losses_dict, accs, gap_accs, sample_acts):
    os.makedirs(out_dir, exist_ok=True)
    # Loss curves
    plt.figure()
    for k, v in losses_dict.items():
        plt.plot(v, label=k)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training loss')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'))
    plt.close()

    # Accuracy summary
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        for name, acc in accs.items():
            f.write(f"{name}: accuracy={acc:.4f}\n")
    # Gap accuracy plot for both models
    plt.figure()
    for name, ga in gap_accs.items():
        gaps = [g for g, a in ga]
        acc = [a for g, a in ga]
        plt.plot(gaps, acc, '-o', label=name)
    plt.xlabel('Gap (positions between memory and query)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs gap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gap_accuracy.png'))
    plt.close()

    # Activation heatmaps for sample
    # sample_acts: dict name -> tensor (positions x dim)
    for name, mat in sample_acts.items():
        plt.figure(figsize=(6, 4))
        plt.imshow(mat, aspect='auto', cmap='bwr')
        plt.colorbar()
        plt.title(f'Activation heatmap: {name}')
        plt.xlabel('Hidden dim')
        plt.ylabel('Position / Block')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'act_{name}.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300, help='Training steps per model')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default=str(DEVICE))
    parser.add_argument('--out', type=str, default='outputs')
    args = parser.parse_args()

    print('Device:', args.device)
    device = torch.device(args.device)

    # Hyperparameters (tiny)
    VOCAB = 64
    SEQ_LEN = 65  # T; history length L = T-1 = 64 which is divisible by block_size=4
    BLOCK = 4
    L = SEQ_LEN - 1

    TRAIN_N = 2000
    TEST_N = 500

    print('Generating dataset...')
    train_x, train_gaps, _ = generate_dataset(TRAIN_N, SEQ_LEN, VOCAB, max_gap=20)
    test_x, test_gaps, _ = generate_dataset(TEST_N, SEQ_LEN, VOCAB, max_gap=20)

    print('Models and training...')
    # Transformer baseline
    tf = TransformerLM(VOCAB, d_model=64, nhead=4, nlayers=2, dim_feedforward=128, max_len=L).to(device)
    velm = VelmProxy(VOCAB, embed_dim=64, latent_dim=48, memory_dim=128, block_size=BLOCK).to(device)

    start = time.time()
    tf_losses = train_model(tf, train_x, train_gaps, steps=args.steps, batch_size=args.batch, lr=1e-3, model_name='transformer', block_size=BLOCK)
    tf_time = time.time() - start
    print(f'Transformer training time: {tf_time:.1f}s')

    start = time.time()
    velm_losses = train_model(velm, train_x, train_gaps, steps=args.steps, batch_size=args.batch, lr=1e-3, cib_lambda=1e-3, model_name='velm', block_size=BLOCK)
    velm_time = time.time() - start
    print(f'VELM training time: {velm_time:.1f}s')

    # Evaluate
    print('Evaluating...')
    tf_acc, tf_gap = evaluate_model(tf, test_x, test_gaps, batch_size=args.batch, model_name='transformer', block_size=BLOCK)
    velm_acc, velm_gap = evaluate_model(velm, test_x, test_gaps, batch_size=args.batch, model_name='velm', block_size=BLOCK)

    print('Transformer acc:', tf_acc)
    print('VELM acc:', velm_acc)

    # Collect sample activations for visualization
    # pick first test batch
    sample = test_x[:args.batch].to(device)
    hist = sample[:, :-1]
    with torch.no_grad():
        tf_logits, tf_attn = tf(hist, return_attn=True)
        velm_states, velm_latents = velm(hist)
    # pick first sample index 0
    tf_acts = tf_logits[0].cpu().numpy()  # (L, V) - use logits as activations
    # for visualization reduce dimension (project vocab dim down)
    tf_acts_small = tf_acts[:, :64]
    velm_acts = velm_states[0].cpu().numpy()  # (n_blocks, hidden)

    out_dir = args.out
    plot_results(out_dir, {'transformer': tf_losses, 'velm': velm_losses}, {'transformer': tf_acc, 'velm': velm_acc}, {'transformer': tf_gap, 'velm': velm_gap}, {'transformer': tf_acts_small, 'velm': velm_acts})

    # Print a short numeric table
    summary_path = os.path.join(out_dir, 'summary.txt')
    print('Wrote outputs to', out_dir)
    with open(summary_path, 'a') as f:
        f.write(f"Transformer time: {tf_time:.1f}s\nVELM time: {velm_time:.1f}s\n")


if __name__ == '__main__':
    main()
