import torch
import torch.nn as nn
import pytest

from velm.model import VelmFull


def test_velmfull_forward_and_step():
    """Run a tiny forward pass and one optimizer step on the PyTorch proxy.

    Ensures shapes, forward, decode, and backward work correctly.
    """
    vocab = 64
    block_size = 4
    L = 16
    batch = 4

    model = VelmFull(vocab_size=vocab, block_size=block_size, embed_dim=16, latent_dim=16, state_dim=32)
    model.train()

    # create synthetic batch where L is divisible by block_size
    x = torch.randint(3, vocab, (batch, L), dtype=torch.long)

    states, latents = model(x, use_adapter=False)

    # expected shapes
    n_blocks = L // block_size
    assert states.shape == (batch, n_blocks, 32)
    assert latents.shape == (batch, n_blocks, 16)

    # decode last block state and run a training step
    b_idx = n_blocks - 1
    state_at = states[:, b_idx, :]
    logits = model.decode_block_state(state_at)  # (B, vocab)
    assert logits.shape == (batch, vocab)

    targets = torch.randint(3, vocab, (batch,), dtype=torch.long)
    loss = nn.functional.cross_entropy(logits, targets)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # loss should be finite
    assert torch.isfinite(loss).all()
