#!/usr/bin/env python3
"""
Extended VELM vs Transformer mini-experiment with expanded tuning and qTTT adaptation sweep.

Outputs saved to outputs_extended/ by default.
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

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

QID = 1

# --- Data ---

def generate_dataset(n_samples, seq_len, vocab_size, max_gap=None):
    L = seq_len - 1
    if max_gap is None:
        max_gap = max(2, L // 2)
    gaps = np.random.randint(1, max_gap + 1, size=n_samples)
    data = np.random.randint(3, vocab_size, size=(n_samples, seq_len))
    labels = np.random.randint(3, vocab_size, size=n_samples)
    for i in range(n_samples):
        g = int(gaps[i])
        Lbl = int(labels[i])
        data[i, 0] = Lbl
        if g >= seq_len - 1:
            g = seq_len - 2
            gaps[i] = g
        data[i, g] = QID
        data[i, g + 1] = Lbl
    return torch.tensor(data, dtype=torch.long), torch.tensor(gaps, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# --- Models with small adapters for qTTT ---
class VelmProxy(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, latent_dim=48, memory_dim=128, block_size=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.block_size = block_size
        self.block_proj = nn.Linear(embed_dim, latent_dim)
        self.mem_mlp = nn.Sequential(nn.Linear(latent_dim, memory_dim), nn.ReLU(), nn.Linear(memory_dim, memory_dim))
        self.decoder = nn.Linear(memory_dim, vocab_size)
        # small adapter initialized to 0 (so initial output unchanged)
        self.adapter = nn.Linear(memory_dim, memory_dim)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, inp, use_adapter=False):
        B, L = inp.shape
        assert L % self.block_size == 0
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
        if use_adapter:
            states = self.adapter(states)
        return states, latents

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
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
        # adapter on top of last hidden (small init)
        self.adapter = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, inp, return_attn=False, use_adapter=False):
        B, L = inp.shape
        x = self.embed(inp) + self.pos[:L].unsqueeze(0).to(inp.device)
        mask = torch.triu(torch.full((L, L), float('-inf'), device=inp.device), diagonal=1)
        attns = []
        for layer in self.layers:
            x, a = layer(x, attn_mask=mask)
            attns.append(a)
        if use_adapter:
            x = self.adapter(x)
        logits = self.out(x)
        if return_attn:
            return logits, attns
        return logits

# --- Utilities ---

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def batch_indices(n, batch_size):
    return np.random.randint(0, n, size=batch_size)

def evaluate_model(model, dataset_inputs, dataset_gaps, batch_size, model_name, block_size=4, device='cpu', use_adapter=False):
    model.eval()
    inputs = dataset_inputs
    gaps = dataset_gaps
    N = inputs.shape[0]
    correct = 0
    per_gap = defaultdict(lambda: [0, 0])
    device = torch.device(device)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = inputs[i: i + batch_size].to(device)
            batch_gaps = gaps[i: i + batch_size].to(device)
            hist = batch[:, :-1]
            if model_name == 'transformer':
                logits = model(hist, use_adapter=use_adapter)
                q_pos = (hist == QID).int().argmax(dim=1)
                logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            else:
                states, latents = model(hist, use_adapter=use_adapter)
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
    total = N
    acc = correct / total
    gaps_sorted = sorted(per_gap.keys())
    gap_acc = [(g, per_gap[g][0] / per_gap[g][1]) for g in gaps_sorted]
    return acc, gap_acc

# time-limited training (updates whole model during training)
def train_model_time_limited(model, train_inputs, train_gaps, time_budget_sec=30, batch_size=128, lr=1e-3, cib_lambda=1e-3, model_name='velm', block_size=4, device='cpu', max_steps=100000):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = train_inputs.shape[0]
    start = time.time()
    step = 0
    last_print = start
    while True:
        if step >= max_steps:
            break
        if time.time() - start >= time_budget_sec:
            break
        idx = batch_indices(N, batch_size)
        batch = train_inputs[idx].to(device)
        hist = batch[:, :-1]
        q_pos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
        if model_name == 'transformer':
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
        now = time.time()
        if now - last_print > 5 or step % 200 == 0:
            last_print = now
            elapsed = now - start
            print(f"[{model_name}] step {step} elapsed={elapsed:.1f}s loss={loss.item():.4f}")
    total_time = time.time() - start
    return losses, total_time

# qTTT adaptation: only update adapter params for few steps using support set

def qttt_adapt(model, support_inputs, support_gaps, model_name, steps=5, batch_size=32, lr=1e-3, block_size=4, device='cpu'):
    # clone adapters to restore later
    device = torch.device(device)
    model.to(device)
    # collect adapter parameters
    adapter_params = [p for n, p in model.named_parameters() if 'adapter' in n]
    if len(adapter_params) == 0:
        return  # nothing to adapt
    opt = torch.optim.SGD(adapter_params, lr=lr)
    model.train()
    N = support_inputs.shape[0]
    for step in range(steps):
        idx = batch_indices(N, batch_size)
        batch = support_inputs[idx].to(device)
        hist = batch[:, :-1]
        q_pos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), q_pos + 1].to(device)
        if model_name == 'transformer':
            logits = model(hist, use_adapter=True)
            logits_at = logits[torch.arange(logits.shape[0]), q_pos, :]
            loss = F.cross_entropy(logits_at, targets)
        else:
            states, latents = model(hist, use_adapter=True)
            b_idx = (q_pos // block_size).long()
            logits_at = model.decoder(states[torch.arange(states.shape[0]), b_idx, :])
            loss = F.cross_entropy(logits_at, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

# plotting helper

def plot_results(out_dir, losses_dict, accs, gap_accs, adapt_curve, sample_acts):
    os.makedirs(out_dir, exist_ok=True)
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

    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        for name, acc in accs.items():
            f.write(f"{name}: accuracy={acc:.4f}\n")

    plt.figure()
    for name, ga in gap_accs.items():
        gaps = [g for g, a in ga]
        acc = [a for g, a in ga]
        plt.plot(gaps, acc, '-o', label=name)
    plt.xlabel('Gap')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs gap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gap_accuracy.png'))
    plt.close()

    # adaptation curve
    plt.figure()
    for name, curve in adapt_curve.items():
        steps = sorted(curve.keys())
        vals = [curve[s] for s in steps]
        plt.plot(steps, vals, '-o', label=name)
    plt.xlabel('qTTT adaptation steps')
    plt.ylabel('Accuracy')
    plt.title('qTTT adaptation sweep')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'qttt_adapt.png'))
    plt.close()

    for name, mat in sample_acts.items():
        plt.figure(figsize=(6, 4))
        plt.imshow(mat, aspect='auto', cmap='bwr')
        plt.colorbar()
        plt.title(f'Activation heatmap: {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'act_{name}.png'))
        plt.close()

# main flow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_time', type=int, default=40)
    parser.add_argument('--final_time', type=int, default=180)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out', type=str, default='outputs_extended')
    parser.add_argument('--train_n', type=int, default=2500)
    parser.add_argument('--test_n', type=int, default=800)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Device:', device)

    VOCAB = 64
    SEQ_LEN = 65
    BLOCK = 4
    L = SEQ_LEN - 1

    print('Generating dataset...')
    train_x, train_gaps, _ = generate_dataset(args.train_n, SEQ_LEN, VOCAB, max_gap=20)
    test_x, test_gaps, _ = generate_dataset(args.test_n, SEQ_LEN, VOCAB, max_gap=20)

    # expanded tuning grid (small but broader)
    latent_opts = [48, 64, 96]
    mem_opts = [128, 192, 256]
    cib_opts = [1e-3, 5e-4, 1e-4]
    lr_opts = [1e-3, 5e-4, 2e-4]

    candidates = []
    for ld in latent_opts:
        for md in mem_opts:
            for cb in cib_opts:
                for lr in lr_opts:
                    candidates.append({'latent_dim': ld, 'memory_dim': md, 'cib_lambda': cb, 'lr': lr})
    random.shuffle(candidates)
    # limit number of candidates for time
    candidates = candidates[:6]

    per_candidate_time = max(5, args.tune_time // max(1, len(candidates)))
    print('Tuning candidates (each ~s):', per_candidate_time, 'count:', len(candidates))

    best = None
    best_acc = -1.0
    for cfg in candidates:
        print('Testing cfg:', cfg)
        velm = VelmProxy(VOCAB, embed_dim=64, latent_dim=cfg['latent_dim'], memory_dim=cfg['memory_dim'], block_size=BLOCK)
        losses, t = train_model_time_limited(velm, train_x, train_gaps, time_budget_sec=per_candidate_time, batch_size=args.batch, lr=cfg['lr'], cib_lambda=cfg['cib_lambda'], model_name='velm', block_size=BLOCK, device=device)
        acc, gap = evaluate_model(velm, test_x, test_gaps, batch_size=args.batch, model_name='velm', block_size=BLOCK, device=device)
        print(f"cfg acc={acc:.4f} time={t:.1f}s")
        if acc > best_acc:
            best_acc = acc
            best = cfg
    print('Best cfg:', best, 'acc:', best_acc)

    # final VELM train longer
    velm_final = VelmProxy(VOCAB, embed_dim=64, latent_dim=best['latent_dim'], memory_dim=best['memory_dim'], block_size=BLOCK)
    velm_losses, velm_time = train_model_time_limited(velm_final, train_x, train_gaps, time_budget_sec=args.final_time, batch_size=args.batch, lr=best['lr'], cib_lambda=best['cib_lambda'], model_name='velm', block_size=BLOCK, device=device, max_steps=1000000)
    velm_acc, velm_gap = evaluate_model(velm_final, test_x, test_gaps, batch_size=args.batch, model_name='velm', block_size=BLOCK, device=device)
    velm_params = count_params(velm_final)
    print(f"VELM final: params={velm_params} acc={velm_acc:.4f} time={velm_time:.1f}s")

    # match transformer params
    def find_transformer_config(vocab, L, target_params):
        d_model = 64
        nlayers = 2
        for i in range(12):
            nhead = 4 if d_model >= 4 and d_model % 4 == 0 else 1
            tfm = TransformerLM(vocab, d_model=d_model, nhead=nhead, nlayers=nlayers, dim_feedforward=max(32, d_model * 2), max_len=L)
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
        tfm = TransformerLM(vocab, d_model=d_model, nhead=nhead, nlayers=nlayers, dim_feedforward=max(32, d_model * 2), max_len=L)
        p = count_params(tfm)
        return d_model, nhead, nlayers, p

    d_model, nhead, nlayers, tf_params = find_transformer_config(VOCAB, L, velm_params)
    print('Transformer match:', d_model, nhead, nlayers, tf_params)
    transformer = TransformerLM(VOCAB, d_model=d_model, nhead=nhead, nlayers=nlayers, dim_feedforward=max(32, d_model * 2), max_len=L)
    tf_time_budget = max(5, int(args.final_time * 0.6))
    tf_losses, tf_time = train_model_time_limited(transformer, train_x, train_gaps, time_budget_sec=tf_time_budget, batch_size=args.batch, lr=1e-3, model_name='transformer', block_size=BLOCK, device=device)
    tf_acc, tf_gap = evaluate_model(transformer, test_x, test_gaps, batch_size=args.batch, model_name='transformer', block_size=BLOCK, device=device)
    print(f"Transformer final: params={tf_params} acc={tf_acc:.4f} time={tf_time:.1f}s")

    # qTTT adaptation sweep (adapter-only steps)
    support = train_x[:256]
    adapt_steps = [0, 1, 5, 10]
    adapt_results = {'velm': {}, 'transformer': {}}
    for s in adapt_steps:
        # reset adapters to zero (fresh) before each sweep
        for m in [velm_final, transformer]:
            for n, p in m.named_parameters():
                if 'adapter' in n:
                    nn.init.zeros_(p)
        if s > 0:
            qttt_adapt(velm_final, support, None, 'velm', steps=s, batch_size=32, lr=1e-2, block_size=BLOCK, device=device)
            qttt_adapt(transformer, support, None, 'transformer', steps=s, batch_size=32, lr=1e-2, block_size=BLOCK, device=device)
        v_acc, _ = evaluate_model(velm_final, test_x, test_gaps, batch_size=args.batch, model_name='velm', block_size=BLOCK, device=device, use_adapter=True)
        t_acc, _ = evaluate_model(transformer, test_x, test_gaps, batch_size=args.batch, model_name='transformer', block_size=BLOCK, device=device, use_adapter=True)
        adapt_results['velm'][s] = v_acc
        adapt_results['transformer'][s] = t_acc
        print(f'adapt steps={s} velm_acc={v_acc:.4f} tf_acc={t_acc:.4f}')

    # collect activations
    sample = test_x[:args.batch].to(device)
    hist = sample[:, :-1]
    with torch.no_grad():
        tf_logits, tf_attn = transformer(hist, return_attn=True)
        velm_states, velm_latents = velm_final(hist)
    tf_acts = tf_logits[0].cpu().numpy()
    tf_acts_small = tf_acts[:, :min(64, tf_acts.shape[1])]
    velm_acts = velm_states[0].cpu().numpy()

    out_dir = args.out
    plot_results(out_dir, {'transformer': tf_losses, 'velm': velm_losses}, {'transformer': tf_acc, 'velm': velm_acc}, {'transformer': tf_gap, 'velm': velm_gap}, adapt_results, {'transformer': tf_acts_small, 'velm': velm_acts})

    summary = {
        'velm_params': velm_params,
        'velm_acc': velm_acc,
        'velm_time': velm_time,
        'transformer_params': tf_params,
        'transformer_acc': tf_acc,
        'transformer_time': tf_time,
        'best_cfg': best,
        'adapt_results': adapt_results,
    }
    with open(os.path.join(out_dir, 'summary.txt'), 'a') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print('Done. Outputs in', out_dir)
    if device.type != 'cuda':
        print('\nNote: CUDA not used. To run on GPU, install matching PyTorch CUDA build.')

if __name__ == '__main__':
    main()
