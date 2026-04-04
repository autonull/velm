#!/usr/bin/env python3
"""
Benchmark Script for VELM vs. Vanilla Transformer on Tiny Shakespeare.
"""
import os
import time
import argparse
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from velm.model import VelmFull

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Data Preparation ---
def get_tiny_shakespeare(block_size=4):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists('input.txt'):
        data = urlopen(url).read().decode('utf-8')
        with open('input.txt', 'w') as f:
            f.write(data)
    else:
        with open('input.txt', 'r') as f:
            data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode data
    data_idx = [stoi[ch] for ch in data]

    # Split: 90% train, 10% val
    n = int(0.9 * len(data_idx))
    train_data = torch.tensor(data_idx[:n], dtype=torch.long)
    val_data = torch.tensor(data_idx[n:], dtype=torch.long)

    # Pre-tokenize to drop remainder
    train_data = train_data[:(len(train_data) // block_size) * block_size]
    val_data = val_data[:(len(val_data) // block_size) * block_size]

    return train_data, val_data, vocab_size, stoi, itos

def get_batch(data, batch_size, seq_len, block_size=4):
    """
    Returns inputs (B, L) and targets (B, L). For VELM we target block decoding.
    seq_len MUST be a multiple of block_size.
    """
    assert seq_len % block_size == 0
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    # Ensure starting indices align with blocks
    ix = (ix // block_size) * block_size

    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# --- Baselines ---

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, nlayers=2, dim_feedforward=128, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(nlayers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.adapter = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, inp, use_adapter=False):
        B, L = inp.shape
        x = self.embed(inp) + self.pos[:L].unsqueeze(0).to(inp.device)
        mask = torch.triu(torch.full((L, L), float('-inf'), device=inp.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        if use_adapter:
            x = self.adapter(x)
        logits = self.out(x)
        return logits

# --- Utilities ---
def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def estimate_loss(model, data, model_name, eval_iters, batch_size, seq_len, block_size, device):
    model.eval()
    losses = []
    accs = []
    cib_norms = []

    for _ in range(eval_iters):
        X, Y = get_batch(data, batch_size, seq_len, block_size)
        X, Y = X.to(device), Y.to(device)

        if model_name == 'velm':
            # VELM is autoregressive across blocks. We want to predict the NEXT block.
            # So state i should predict block i+1.
            # To do this, we provide history up to N-1, and predict 1 to N.

            # Shift X and Y
            # For autoregressive block prediction, we can just encode the whole X,
            # and use state[:, :-1] to predict Y blocks[:, 1:]

            states, latents = model(X) # states: (B, N, state_dim), latents: (B, N, latent_dim)
            B, N, _ = states.shape
            K = block_size

            # We use state[i] to predict block i+1. So we take states[:, :-1]
            # and target blocks Y[:, 1:]
            # But wait, Y already represents the shifted targets (next token).
            # Y shape is (B, N*K). X is x[t], Y is x[t+1].
            # Actually, standard LM targets Y are just X shifted by 1.
            # In block-level autoregression:
            # state i encodes X blocks 0..i.
            # It should predict the next K tokens following X block i.
            # Those tokens are exactly Y block i!
            # Because X block i is [x_{i*K}, ..., x_{i*K+K-1}],
            # Y block i is [x_{i*K+1}, ..., x_{i*K+K}].
            # This is standard teacher-forcing.

            states_flat = states.view(B * N, -1)
            logits = model.decode_block_state(states_flat) # (B*N, K, V)

            targets = Y.view(B * N, K)

            # Flatten to compute loss across all tokens
            logits_flat = logits.view(-1, logits.size(-1)) # (B*N*K, V)
            targets_flat = targets.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat)

            preds = logits_flat.argmax(dim=-1)
            acc = (preds == targets_flat).float().mean().item()

            cib = latents.norm(p=2, dim=2).mean().item()
            cib_norms.append(cib)
        else:
            logits = model(X)
            B, L, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets = Y.view(-1)
            loss = F.cross_entropy(logits_flat, targets)

            preds = logits_flat.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

    model.train()
    if model_name == 'velm':
        return np.mean(losses), np.mean(accs), np.mean(cib_norms)
    return np.mean(losses), np.mean(accs), None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='benchmark_outputs')
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128) # divisible by block_size
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_data, val_data, vocab_size, stoi, itos = get_tiny_shakespeare(args.block_size)
    print(f"Vocab size: {vocab_size}, Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Instantiate VELM scaled up to match Transformer
    velm = VelmFull(vocab_size=vocab_size, block_size=args.block_size, embed_dim=128, latent_dim=128, state_dim=192)
    velm_params = count_params(velm)

    # Instantiate Transformer with roughly similar params
    # ~ Velm = 128*64 + 64*128*2 + ... ~ small.
    transformer = TransformerLM(vocab_size=vocab_size, d_model=128, nhead=4, nlayers=2, dim_feedforward=256, max_len=args.seq_len)
    tf_params = count_params(transformer)

    print(f"VELM params: {velm_params}")
    print(f"Transformer params: {tf_params}")

    models = {
        'transformer': transformer,
        'velm': velm
    }

    results = {'velm': {'train_loss':[], 'val_loss':[], 'val_acc':[], 'cib_norm':[]},
               'transformer': {'train_loss':[], 'val_loss':[], 'val_acc':[]}}

    for name, model in models.items():
        print(f"\\n--- Training {name.upper()} ---")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        t0 = time.time()
        t_start_step = time.time()
        for iter_num in range(args.max_iters):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
                train_loss, train_acc, cib_norm = estimate_loss(model, train_data, name, args.eval_iters, args.batch_size, args.seq_len, args.block_size, device)
                val_loss, val_acc, val_cib_norm = estimate_loss(model, val_data, name, args.eval_iters, args.batch_size, args.seq_len, args.block_size, device)

                results[name]['train_loss'].append(train_loss)
                results[name]['val_loss'].append(val_loss)
                results[name]['val_acc'].append(val_acc)
                if cib_norm is not None:
                    results[name]['cib_norm'].append(val_cib_norm)

                t_end_step = time.time()
                steps_per_sec = args.eval_interval / (t_end_step - t_start_step) if iter_num > 0 else 0
                t_start_step = time.time()

                print(f"Step {iter_num}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}" +
                      (f", CIB Norm {val_cib_norm:.4f}" if cib_norm is not None else "") +
                      (f", Speed {steps_per_sec:.2f} steps/s" if iter_num > 0 else ""))

            X, Y = get_batch(train_data, args.batch_size, args.seq_len, args.block_size)
            X, Y = X.to(device), Y.to(device)

            if name == 'velm':
                states, latents = model(X)
                B, N, _ = states.shape
                K = args.block_size

                states_flat = states.view(B * N, -1)
                logits = model.decode_block_state(states_flat) # (B*N, K, V)

                targets = Y.view(B * N, K)

                logits_flat = logits.view(-1, logits.size(-1)) # (B*N*K, V)
                targets_flat = targets.view(-1)

                loss = F.cross_entropy(logits_flat, targets_flat)

                # CIB Loss
                cib_loss = 1e-3 * latents.norm(p=2, dim=2).mean()
                loss = loss + cib_loss
            else:
                logits = model(X)
                logits_flat = logits.view(-1, vocab_size)
                targets = Y.view(-1)
                loss = F.cross_entropy(logits_flat, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        t1 = time.time()
        print(f"{name.upper()} Training Time: {t1-t0:.2f}s")

    # Plotting

    # 1. Loss Comparison
    plt.figure()
    for name in models:
        plt.plot(np.arange(len(results[name]['train_loss'])) * args.eval_interval, results[name]['train_loss'], '--', label=f"{name} train loss", alpha=0.7)
        plt.plot(np.arange(len(results[name]['val_loss'])) * args.eval_interval, results[name]['val_loss'], label=f"{name} val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Comparison")
    plt.legend()
    plt.savefig(os.path.join(args.out, "loss_comparison.png"))
    plt.close()

    # 2. Accuracy Comparison
    plt.figure()
    for name in models:
        plt.plot(np.arange(len(results[name]['val_acc'])) * args.eval_interval, results[name]['val_acc'], label=f"{name} val acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.savefig(os.path.join(args.out, "accuracy_comparison.png"))
    plt.close()

    # 3. CIB Norm (VELM only)
    if len(results['velm']['cib_norm']) > 0:
        plt.figure()
        plt.plot(np.arange(len(results['velm']['cib_norm'])) * args.eval_interval, results['velm']['cib_norm'], label="VELM CIB norm", color='green')
        plt.xlabel("Step")
        plt.ylabel("Norm")
        plt.title("VELM Continuous Information Bottleneck Norm")
        plt.legend()
        plt.savefig(os.path.join(args.out, "cib_norm.png"))
        plt.close()

    # --- qTTT (Test-Time Training) Demonstration for VELM ---
    print("\\n--- Running qTTT Adaptation Demonstration on VELM ---")

    # We take a single long validation chunk to act as a "document".
    # We split it into support (for adapting the adapter) and query (for testing).

    # Context must be evenly divisible by block_size
    document_len = 512
    support_len = 256
    query_len = 256

    # Ensure val_data is long enough to grab document + 1 for offset target
    start_idx = torch.randint(0, len(val_data) - document_len - 1, (1,)).item()
    start_idx = (start_idx // args.block_size) * args.block_size

    document_x = val_data[start_idx:start_idx+document_len].to(device)
    document_y = val_data[start_idx+1:start_idx+document_len+1].to(device)

    support_x = document_x[:support_len].unsqueeze(0) # (1, 256)
    support_y = document_y[:support_len].unsqueeze(0)

    query_x = document_x[support_len:support_len+query_len].unsqueeze(0) # (1, 256)
    query_y = document_y[support_len:support_len+query_len].unsqueeze(0)

    def eval_chunk(model, x, y):
        model.eval()
        with torch.no_grad():
            states, latents = model(x, use_adapter=True)
            B, N, _ = states.shape
            K = args.block_size
            states_flat = states.view(B * N, -1)
            logits = model.decode_block_state(states_flat)
            targets = y.view(B * N, K)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            preds = logits_flat.argmax(dim=-1)
            acc = (preds == targets_flat).float().mean().item()
            return acc

    # 1. Zero-shot baseline on the query chunk (adapter is initialized to identity/zero)
    velm.eval()
    zero_shot_acc = eval_chunk(velm, query_x, query_y)

    # 2. Adapt only the adapter weights on the support chunk
    adapter_params = [p for n, p in velm.named_parameters() if 'adapter' in n]
    qttt_opt = torch.optim.AdamW(adapter_params, lr=1e-2)

    velm.train()
    adapt_steps = 25

    for step in range(adapt_steps):
        states, latents = velm(support_x, use_adapter=True)
        B, N, _ = states.shape
        K = args.block_size
        states_flat = states.view(B * N, -1)
        logits = velm.decode_block_state(states_flat)
        targets = support_y.view(B * N, K)

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)

        qttt_opt.zero_grad()
        loss.backward()
        qttt_opt.step()

    # 3. Re-evaluate on query chunk with adapted weights
    velm.eval()
    adapted_acc = eval_chunk(velm, query_x, query_y)

    print(f"qTTT Zero-shot Query Acc: {zero_shot_acc:.4f}")
    print(f"qTTT Adapted Query Acc (after {adapt_steps} support steps): {adapted_acc:.4f}")

    # Save Results summary
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(f"Vocab size: {vocab_size}\\n")
        f.write(f"VELM params: {velm_params}\\n")
        f.write(f"Transformer params: {tf_params}\\n\\n")
        for name in models:
            f.write(f"{name.upper()} Final Val Loss: {results[name]['val_loss'][-1]:.4f}\\n")
            f.write(f"{name.upper()} Final Val Acc: {results[name]['val_acc'][-1]:.4f}\\n")
            if name == 'velm':
                f.write(f"{name.upper()} Final CIB Norm: {results[name]['cib_norm'][-1]:.4f}\\n")

        f.write(f"\\n--- qTTT Experiment ---\\n")
        f.write(f"Zero-shot Acc: {zero_shot_acc:.4f}\\n")
        f.write(f"Adapted Acc: {adapted_acc:.4f}\\n")

    print(f"Results saved to {args.out}/")

if __name__ == '__main__':
    main()
