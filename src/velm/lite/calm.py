import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CALMEncoder(nn.Module):
    """Compress K tokens into one continuous vector.
    Adaptive K support: Pads smaller chunks to max block_size before MLP projection.
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Use SwiGLU for enhanced representation efficiency
        self.proj = SwiGLU(embed_dim * block_size, latent_dim * 2, latent_dim)

    def forward(self, tokens):
        # tokens: (B, K)
        B, K = tokens.shape
        emb = self.embed(tokens)  # (B, K, E)

        # Adaptive chunk size: pad with zeros if K < block_size
        if K < self.block_size:
            pad_len = self.block_size - K
            pad_tensor = torch.zeros(B, pad_len, emb.size(-1), device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad_tensor], dim=1) # (B, block_size, E)
        elif K > self.block_size:
            # truncate if larger (should not happen in proper usage)
            emb = emb[:, :self.block_size, :]

        flat = emb.view(B, self.block_size * emb.size(-1)) # (B, max_K*E)
        lat = self.proj(flat)     # (B, latent)
        return lat

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        # Use SwiGLU for enhanced representation efficiency
        self.net = SwiGLU(dim, hidden, dim)

    def forward(self, x):
        return x + self.net(x)

class CALMDecoder(nn.Module):
    """Energy-based style decoder stub: map latent to K tokens logits via MLP with residual blocks."""
    def __init__(self, latent_dim=64, hidden=128, vocab_size=100, block_size=4, num_blocks=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.blocks = nn.Sequential(*[ResidualBlock(latent_dim, hidden) for _ in range(num_blocks)])
        self.head = nn.Linear(latent_dim, vocab_size * block_size)

    def forward(self, latent, target_K=None):
        # latent: (..., latent_dim)
        # target_K: optional dynamic sequence length to return.
        # out: (..., K, vocab_size)
        h = self.blocks(latent)
        out = self.head(h)
        shape = list(out.shape[:-1]) + [self.block_size, self.vocab_size]
        out = out.view(*shape)

        # Adaptive chunk size support: Truncate output to target_K if specified
        if target_K is not None and target_K < self.block_size:
            out = out[..., :target_K, :]
        return out
