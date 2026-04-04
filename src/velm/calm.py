import torch
import torch.nn as nn
import torch.nn.functional as F

class CALMEncoder(nn.Module):
    """Compress K tokens into one continuous vector (simple proxy).
    Implementation: embed tokens, flatten to preserve order, then MLP projection.
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * block_size, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, tokens):
        # tokens: (B, K)
        B, K = tokens.shape
        emb = self.embed(tokens)  # (B, K, E)
        flat = emb.view(B, K * emb.size(-1)) # (B, K*E)
        lat = self.proj(flat)     # (B, latent)
        return lat

class CALMDecoder(nn.Module):
    """Energy-based style decoder stub: map latent to K tokens logits via MLP."""
    def __init__(self, latent_dim=64, hidden=128, vocab_size=100, block_size=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size * block_size)
        )

    def forward(self, latent):
        # latent: (..., latent_dim)
        # out: (..., block_size, vocab_size)
        out = self.net(latent)
        shape = list(out.shape[:-1]) + [self.block_size, self.vocab_size]
        return out.view(*shape)
