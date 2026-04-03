import torch
import torch.nn as nn
import torch.nn.functional as F

class CALMEncoder(nn.Module):
    """Compress K tokens into one continuous vector (simple proxy).
    Implementation: embed tokens, elementwise mean, then MLP projection.
    """
    def __init__(self, vocab_size, embed_dim=64, latent_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))

    def forward(self, tokens):
        # tokens: (B, K)
        emb = self.embed(tokens)  # (B, K, E)
        mean = emb.mean(dim=1)    # (B, E)
        lat = self.proj(mean)     # (B, latent)
        return lat

class CALMDecoder(nn.Module):
    """Energy-based style decoder stub: map latent to token logits via MLP."""
    def __init__(self, latent_dim=64, hidden=128, vocab_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size))

    def forward(self, latent):
        return self.net(latent)
