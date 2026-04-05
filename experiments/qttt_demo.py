#!/usr/bin/env python3
"""qTTT demo: adapt Adapter modules only for a few steps on a small held-out batch and show pre/post accuracy."""

import os
import sys
from pathlib import Path

# Ensure src/ is on the path so `velm.*` imports work when running standalone
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import torch.nn.functional as F
from velm.lite import VelmFull

QID = 1


def generate_dataset(n, seq_len, vocab, max_gap=8):
    L = seq_len - 1
    gaps = torch.randint(1, max_gap + 1, (n,), dtype=torch.long)
    data = torch.randint(3, vocab, (n, seq_len), dtype=torch.long)
    for i in range(n):
        g = int(gaps[i].item())
        lbl = int(torch.randint(3, vocab, (1,)).item())
        data[i, 0] = lbl
        if g >= seq_len - 1:
            g = seq_len - 2
        data[i, g] = QID
        data[i, g + 1] = lbl
    return data, gaps


def eval_accuracy(model, data, n_eval=128, device="cpu"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, data.shape[0], (min(n_eval, data.shape[0]),))
        batch = data[idx].to(device)
        hist = batch[:, :-1]
        qpos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), qpos + 1].to(device)
        states, _ = model(hist, use_adapter=True)
        b_idx = (qpos // model.block_size).long()
        state_at = states[torch.arange(batch.shape[0]), b_idx, :]
        logits = model.decode_block_state(state_at)
        preds = logits[:, -1, :].argmax(dim=1)  # predict last token of the block
        acc = (preds == targets).float().mean().item()
    model.train()
    return acc


def adapt_adapter(model, data, steps=20, batch_size=32, lr=1e-3, device="cpu"):
    # freeze all params except adapter
    for n, p in model.named_parameters():
        p.requires_grad = "adapter" in n
    adapter_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam(adapter_params, lr=lr)
    model.to(device)
    N = data.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(batch_size, N),))
        batch = data[idx].to(device)
        hist = batch[:, :-1]
        qpos = (hist == QID).int().argmax(dim=1)
        targets = batch[torch.arange(batch.shape[0]), qpos + 1].to(device)
        states, _ = model(hist, use_adapter=True)
        b_idx = (qpos // model.block_size).long()
        state_at = states[torch.arange(batch.shape[0]), b_idx, :]
        logits = model.decode_block_state(state_at)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/qttt_demo")
    parser.add_argument(
        "--smoketest", action="store_true", help="Quick smoke test: 5 adaptation steps"
    )
    args = parser.parse_args()

    steps = 5 if args.smoketest else 40
    os.makedirs(args.out, exist_ok=True)

    vocab = 128
    seq_len = 33
    data, gaps = generate_dataset(600, seq_len, vocab, max_gap=8)
    model = VelmFull(vocab_size=vocab, block_size=4, embed_dim=32, latent_dim=48, state_dim=96)

    pre = eval_accuracy(model, data, n_eval=256, device="cpu")
    adapt_adapter(model, data, steps=steps, batch_size=64, lr=5e-4, device="cpu")
    post = eval_accuracy(model, data, n_eval=256, device="cpu")

    with open(os.path.join(args.out, "qttt_prepost.txt"), "w") as f:
        f.write(f"pre_acc={pre}\npost_acc={post}\n")
    print(f"qTTT demo complete. pre_acc={pre:.4f} post_acc={post:.4f} delta={post - pre:+.4f}")


if __name__ == "__main__":
    main()
