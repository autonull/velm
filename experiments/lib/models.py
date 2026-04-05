"""Shared model definitions for VELM experiments.

Contains:
  - VelmProxy: simplified VELM (block compression + MLP memory)
  - TransformerLM: causal transformer baseline
  - Utilities: count_params, find_transformer_config
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QID = 1


# ---------------------------------------------------------------------------
# VELM Proxy
# ---------------------------------------------------------------------------


class VelmProxy(nn.Module):
    """Tiny VELM proxy: CALM-like block compression + simple MLP memory + linear decoder."""

    def __init__(self, vocab_size, embed_dim=64, latent_dim=48, memory_dim=128, block_size=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.block_size = block_size
        self.block_proj = nn.Linear(embed_dim, latent_dim)
        self.mem_mlp = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.decoder = nn.Linear(memory_dim, vocab_size)

    def forward(self, inp):
        B, L = inp.shape
        assert L % self.block_size == 0, "History length must be divisible by block_size"
        n_blocks = L // self.block_size
        emb = self.embed(inp)
        emb_blocks = emb.view(B, n_blocks, self.block_size, -1)
        block_mean = emb_blocks.mean(dim=2)
        latents = self.block_proj(block_mean)
        memory_dim = self.mem_mlp[-1].out_features
        state = torch.zeros(B, memory_dim, device=latents.device)
        states = []
        for i in range(n_blocks):
            delta = self.mem_mlp(latents[:, i, :])
            state = state + delta
            states.append(state.unsqueeze(1))
        states = torch.cat(states, dim=1)
        return states, latents


# ---------------------------------------------------------------------------
# Transformer Baseline
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_out, attn_w = self.self_attn(
            x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False
        )
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x, attn_w


class TransformerLM(nn.Module):
    """Causal transformer baseline."""

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

    def forward(self, inp, return_attn=False):
        B, L = inp.shape
        x = self.embed(inp) + self.pos[:L].unsqueeze(0).to(inp.device)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=inp.device), diagonal=1)
        attns = []
        for layer in self.layers:
            x, a = layer(x, attn_mask=mask)
            attns.append(a)
        logits = self.out(x)
        if return_attn:
            return logits, attns
        return logits


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_transformer_config(vocab, L, target_params, init_d_model=64, init_nlayers=2):
    """Find transformer config whose param count is close to target_params."""
    d_model = init_d_model
    nlayers = init_nlayers
    for _ in range(12):
        nhead = 4 if d_model >= 4 and d_model % 4 == 0 else 1
        tfm = TransformerLM(
            vocab,
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
            dim_feedforward=max(32, d_model * 2),
            max_len=L,
        )
        p = count_params(tfm)
        if abs(p - target_params) / max(1, target_params) < 0.12:
            return d_model, nhead, nlayers, p
        scale = math.sqrt(target_params / max(1, p))
        new_d = max(8, int(d_model * scale))
        if new_d == d_model:
            if p < target_params and nlayers < 6:
                nlayers += 1
            elif p > target_params and nlayers > 1:
                nlayers -= 1
            else:
                new_d = d_model + (1 if p < target_params else -1)
        d_model = max(8, min(512, new_d))
    nhead = 4 if d_model >= 4 and d_model % 4 == 0 else 1
    tfm = TransformerLM(
        vocab,
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dim_feedforward=max(32, d_model * 2),
        max_len=L,
    )
    p = count_params(tfm)
    return d_model, nhead, nlayers, p


def evaluate_model(
    model, dataset_inputs, dataset_gaps, batch_size, model_name, block_size=4, device="cpu"
):
    """Evaluate model on delayed-recall task. Returns (accuracy, per-gap-accuracy)."""
    model.eval()
    N = dataset_inputs.shape[0]
    correct = 0
    per_gap = defaultdict(lambda: [0, 0])
    device = torch.device(device)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = dataset_inputs[i : i + batch_size].to(device)
            batch_gaps = dataset_gaps[i : i + batch_size].to(device)
            hist = batch[:, :-1]
            if model_name == "transformer":
                logits = model(hist)
                q_pos = (hist == QID).int().argmax(dim=1)
                logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            else:
                states, _ = model(hist)
                q_pos = (hist == QID).int().argmax(dim=1)
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
    acc = correct / N
    gaps_sorted = sorted(per_gap.keys())
    gap_acc = [(g, per_gap[g][0] / per_gap[g][1]) for g in gaps_sorted]
    return acc, gap_acc


def train_model(
    model,
    train_inputs,
    train_gaps,
    steps=300,
    batch_size=128,
    lr=1e-3,
    cib_lambda=1e-3,
    model_name="transformer",
    block_size=4,
    device="cpu",
):
    """Train model for a fixed number of steps. Returns list of losses."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = train_inputs.shape[0]
    for step in range(1, steps + 1):
        idx = np.random.randint(0, N, size=batch_size)
        batch = train_inputs[idx].to(device)
        batch_gaps = train_gaps[idx].to(device)
        hist = batch[:, :-1]
        q_pos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
        if model_name == "transformer":
            logits = model(hist)
            logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            loss = F.cross_entropy(logits_at, targets)
        else:
            states, latents = model(hist)
            b_idx = (q_pos // block_size).long()
            logits_at = model.decoder(states[torch.arange(states.shape[0]), b_idx, :])
            loss = F.cross_entropy(logits_at, targets)
            lat_q = latents[torch.arange(latents.shape[0]), b_idx, :]
            cib = lat_q.norm(p=2, dim=1).mean()
            loss = loss + cib_lambda * cib
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, steps // 10) == 0 or step <= 5:
            print(f"  [{model_name}] step {step}/{steps} loss={loss.item():.4f}")
    return losses


def train_model_time_limited(
    model,
    train_inputs,
    train_gaps,
    time_budget_sec=30,
    batch_size=128,
    lr=1e-3,
    cib_lambda=1e-3,
    model_name="transformer",
    block_size=4,
    device="cpu",
    max_steps=100000,
):
    """Train model for a fixed time budget. Returns (losses, elapsed_time)."""
    import time as _time

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = train_inputs.shape[0]
    start = _time.time()
    step = 0
    last_print = start
    while step < max_steps and (_time.time() - start) < time_budget_sec:
        idx = np.random.randint(0, N, size=batch_size)
        batch = train_inputs[idx].to(device)
        hist = batch[:, :-1]
        q_pos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
        if model_name == "transformer":
            logits = model(hist)
            logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            loss = F.cross_entropy(logits_at, targets)
        else:
            states, latents = model(hist)
            b_idx = (q_pos // block_size).long()
            logits_at = model.decoder(states[torch.arange(states.shape[0]), b_idx, :])
            loss = F.cross_entropy(logits_at, targets)
            lat_q = latents[torch.arange(latents.shape[0]), b_idx, :]
            cib = lat_q.norm(p=2, dim=1).mean()
            loss = loss + cib_lambda * cib
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
        now = _time.time()
        if now - last_print > 5 or step % 100 == 0:
            last_print = now
            elapsed = now - start
            print(f"  [{model_name}] step {step} elapsed={elapsed:.1f}s loss={loss.item():.4f}")
    return losses, _time.time() - start
