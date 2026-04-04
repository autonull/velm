import time
import pytest

pytestmark = pytest.mark.benchmark

from velm.torch_proxy import VelmFull
import torch


def test_torch_forward_benchmark():
    """Simple micro-benchmark to measure PyTorch proxy forward throughput.

    Marked with pytest benchmark marker so CI can include/exclude.
    """
    model = VelmFull(vocab_size=256, block_size=4, embed_dim=32, latent_dim=32, state_dim=64)
    model.eval()

    batch = 8
    L = 32
    x = torch.randint(3, 256, (batch, L), dtype=torch.long)

    # warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(x)

    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
    dt = (time.perf_counter() - t0) / iters

    # ensure per-iteration time is reasonable on CI (very generous threshold)
    assert dt < 5.0
