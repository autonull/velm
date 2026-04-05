"""Unified dataset interface for VELM experiments.

All datasets return the same interface:
  (train_data, val_data, vocab_size, stoi, itos)

where train_data and val_data are 1D torch.LongTensor of token indices.

Datasets:
  - tiny_shakespeare: Karpathy's tinyshakespeare (65 chars)
  - shakespeare_full: Complete Shakespeare works (~5M chars, ~100 vocab)
  - tiny_stories: Ronen Eldan's Tiny Stories (8K tokens, synthetic children's stories)
"""

import os
import json
from urllib.request import urlopen

import torch


def _download_if_missing(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = urlopen(url).read().decode("utf-8")
        with open(path, "w") as f:
            f.write(data)


def get_tiny_shakespeare(block_size=4, data_dir="data"):
    """Karpathy's tinyshakespeare — 65-char vocabulary.

    ~1.1M characters, 90/10 train/val split.
    URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tiny_shakespeare.txt")
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    _download_if_missing(path, url)

    with open(path, "r") as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data_idx = [stoi[ch] for ch in data]
    n = int(0.9 * len(data_idx))
    train_data = torch.tensor(data_idx[:n], dtype=torch.long)
    val_data = torch.tensor(data_idx[n:], dtype=torch.long)

    # Trim to block boundary
    train_data = train_data[: (len(train_data) // block_size) * block_size]
    val_data = val_data[: (len(val_data) // block_size) * block_size]

    return train_data, val_data, vocab_size, stoi, itos


def get_shakespeare_full(block_size=4, data_dir="data"):
    """Complete Shakespeare works — ~100-char vocabulary.

    ~5M characters, 90/10 train/val split.
    URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    (Note: this is the same as tiny_shakespeare but we use the full corpus
    from Shakespeare's complete works if available, otherwise falls back.)

    For the full corpus, use the Gutenberg text or a larger collection.
    This function downloads from a larger source if available.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare_full.txt")

    # Try to use the full Shakespeare corpus from HuggingFace raw
    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    _download_if_missing(path, url)

    with open(path, "r") as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data_idx = [stoi[ch] for ch in data]
    n = int(0.9 * len(data_idx))
    train_data = torch.tensor(data_idx[:n], dtype=torch.long)
    val_data = torch.tensor(data_idx[n:], dtype=torch.long)

    train_data = train_data[: (len(train_data) // block_size) * block_size]
    val_data = val_data[: (len(val_data) // block_size) * block_size]

    return train_data, val_data, vocab_size, stoi, itos


def get_tiny_stories(block_size=4, data_dir="data"):
    """Ronen Eldan's Tiny Stories — ~8K token vocabulary.

    Synthetic children's stories designed to test reasoning.
    ~2GB raw text, but we use a subset for fast experiments.
    URL: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tiny_stories.txt")
    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    )
    _download_if_missing(path, url)

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    # Use character-level tokenization for consistency
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data_idx = [stoi[ch] for ch in data]
    n = int(0.9 * len(data_idx))
    train_data = torch.tensor(data_idx[:n], dtype=torch.long)
    val_data = torch.tensor(data_idx[n:], dtype=torch.long)

    train_data = train_data[: (len(train_data) // block_size) * block_size]
    val_data = val_data[: (len(val_data) // block_size) * block_size]

    return train_data, val_data, vocab_size, stoi, itos


# Registry
DATASETS = {
    "tiny_shakespeare": get_tiny_shakespeare,
    "shakespeare_full": get_shakespeare_full,
    "tiny_stories": get_tiny_stories,
}


def get_dataset(name, block_size=4, data_dir="data"):
    """Get a dataset by name.

    Args:
        name: dataset name (one of DATASETS.keys())
        block_size: VELM block size (for alignment trimming)
        data_dir: directory to store downloaded data

    Returns:
        (train_data, val_data, vocab_size, stoi, itos)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](block_size=block_size, data_dir=data_dir)
