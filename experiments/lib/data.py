"""Shared data generation for VELM experiments.

All experiments use the same synthetic delayed-recall task:
  - A label token is placed at position 0
  - A query marker (QID=1) is placed at a random gap position
  - The model must predict the label token after seeing the query
  - This tests whether the model can retain information over variable distances
"""

import numpy as np
import torch

QID = 1


def generate_delayed_recall(n_samples, seq_len, vocab_size, max_gap=None, seed=None):
    """Generate a delayed-recall dataset.

    Each sample: [label, noise..., Q, label, noise...]
    The model sees tokens up to (but not including) the second label occurrence,
    and must predict the label that follows the query marker.

    Args:
        n_samples: number of samples
        seq_len: total sequence length
        vocab_size: vocabulary size (tokens 3..vocab_size-1 are noise)
        max_gap: maximum gap between label and query (default: seq_len//2)
        seed: optional random seed for reproducibility

    Returns:
        (data, gaps, labels) tensors of shape (n_samples, seq_len), (n_samples,), (n_samples,)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42)

    L = seq_len - 1
    if max_gap is None:
        max_gap = max(2, L // 2)

    gaps = rng.randint(1, max_gap + 1, size=n_samples)
    data = rng.randint(3, vocab_size, size=(n_samples, seq_len))
    labels = rng.randint(3, vocab_size, size=n_samples)

    for i in range(n_samples):
        g = int(gaps[i])
        lbl = int(labels[i])
        data[i, 0] = lbl  # memory token at position 0
        if g >= seq_len - 1:
            g = seq_len - 2
            gaps[i] = g
        data[i, g] = QID  # query marker
        data[i, g + 1] = lbl  # target token after query

    return (
        torch.tensor(data, dtype=torch.long),
        torch.tensor(gaps, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
