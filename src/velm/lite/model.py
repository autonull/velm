import torch.nn.functional as F
import torch
import torch.nn as nn
from .calm import CALMEncoder, CALMDecoder, SwiGLU
from .miras import MirasMemoryLayer
from .swa import SWALayer, RMSNorm
from .qttt import Adapter

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate, bias=False)
        self.w_up = nn.Linear(dim, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MirasBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_intermediate):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.miras = MirasMemoryLayer(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate)

    def forward(self, x, state=None):
        normed = self.norm1(x)
        miras_out, new_state = self.miras(normed, state)
        x = x + miras_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x, new_state

class SWABlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_intermediate, window_size=32):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SWALayer(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_intermediate)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        return x

class VelmHybrid(nn.Module):
    """Hybrid VELM model combining best of VELM Full and Lite.
    Uses CALM autoencoder block, interleaved Miras (Memora variant) and SWA blocks.
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64, state_dim=128, num_miras_layers=1, num_swa_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_dim = state_dim

        # CALM Encoder
        self.encoder = CALMEncoder(vocab_size, block_size=block_size, embed_dim=embed_dim, latent_dim=latent_dim)

        # Input compression mapping latent to state_dim (acts like backbone compress)
        self.input_compress = nn.Linear(latent_dim, state_dim, bias=False)
        self.input_ffn = SwiGLUFFN(state_dim, state_dim * 2)

        # Blocks (Samba-style interleaved)
        self.blocks = nn.ModuleList()
        self.block_types = []

        num_heads = max(1, state_dim // 32)
        m_idx, s_idx = 0, 0
        total_layers = num_miras_layers + num_swa_layers

        for i in range(total_layers):
            if i % 2 == 0 and m_idx < num_miras_layers:
                self.blocks.append(MirasBlock(state_dim, num_heads, state_dim * 2))
                self.block_types.append("miras")
                m_idx += 1
            elif s_idx < num_swa_layers:
                self.blocks.append(SWABlock(state_dim, num_heads, state_dim * 2))
                self.block_types.append("swa")
                s_idx += 1
            else:
                if m_idx < num_miras_layers:
                    self.blocks.append(MirasBlock(state_dim, num_heads, state_dim * 2))
                    self.block_types.append("miras")
                    m_idx += 1
                else:
                    self.blocks.append(SWABlock(state_dim, num_heads, state_dim * 2))
                    self.block_types.append("swa")
                    s_idx += 1

        self.final_norm = RMSNorm(state_dim)

        self.adapter = Adapter(state_dim, bottleneck=max(8, state_dim//4))

        # CALM Decoder maps state back to tokens
        self.decoder = CALMDecoder(latent_dim=state_dim, hidden=state_dim, vocab_size=vocab_size, block_size=block_size)

    def forward(self, history, use_adapter=False, return_cib_loss=False):
        B, L = history.shape
        K = self.block_size

        # Adaptive K sequence chunking: allow sequences not divisible by block_size
        latents = []
        k_logits_list = [] if return_cib_loss else None

        start_idx = 0
        while start_idx < L:
            # We predict the chunk size dynamically up to block_size.
            # In an actual autoregressive decoding loop, this allows the model
            # to choose how many tokens to process at once.
            chunk = history[:, start_idx:start_idx+K]
            if return_cib_loss:
                lat, k_logits = self.encoder(chunk, return_k_logits=True)
                k_logits_list.append(k_logits.unsqueeze(1))
                # For training purposes in proxy, we process fixed chunks and predict K.
                # In full usage, K = k_logits.argmax(dim=-1).item()
            else:
                lat = self.encoder(chunk)

            latents.append(lat.unsqueeze(1))
            start_idx += K

        latents = torch.cat(latents, dim=1) # (B, n_blocks, latent)

        # Compress latent to state
        x = self.input_ffn(self.input_compress(latents)) # (B, n_blocks, state_dim)

        # Backbone forward
        states = []
        for i, b_type in enumerate(self.block_types):
            if b_type == "miras":
                x, _ = self.blocks[i](x)
            else:
                x = self.blocks[i](x)

        x = self.final_norm(x)

        if use_adapter:
            x = self.adapter(x)

        if return_cib_loss:
            # Conditional Information Bottleneck (CIB) heuristic loss
            cib_loss = latents.norm(p=2, dim=2).mean()
            k_logits_tensor = torch.cat(k_logits_list, dim=1) if k_logits_list else None
            return x, latents, cib_loss, k_logits_tensor

        return x, latents

    def decode_block_state(self, state_block, target_K=None):
        return self.decoder(state_block, target_K=target_K)

class VelmCore(nn.Module):
    """Core VELM sequential backbone.

    Flow: continuous latents -> Miras memory -> optional SWA -> optional adapter
    This isolates the continuous sequence modeling from the specific input/output modalities.
    """
    def __init__(self, latent_dim=64, state_dim=128, use_swa=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.use_swa = use_swa

        self.latent_norm = RMSNorm(latent_dim)
        from .miras import MirasMemory
        self.memory = MirasMemory(key_dim=latent_dim, state_dim=state_dim)

        if self.use_swa:
            self.swa_norm = RMSNorm(state_dim)
            self.swa = SWALayer(dim=state_dim, num_heads=max(1, state_dim // 32), window_size=32)

        self.adapter = Adapter(state_dim, bottleneck=max(8, state_dim//4))

    def forward(self, latents, use_adapter=False):
        # latents: (B, n_blocks, latent_dim)
        norm_latents = self.latent_norm(latents)
        states = self.memory(norm_latents)        # (B, n_blocks, state_dim)

        # Hybrid configuration: SWA provides local context, Miras provides long-range state
        if self.use_swa:
            # Pre-LN Residual connection around SWA for stability
            norm_states = self.swa_norm(states)
            states = states + self.swa(norm_states)

        # optionally apply adapter to all states (qTTT)
        if use_adapter:
            states = self.adapter(states)
        return states

class VelmFull(nn.Module):
    """Full VELM-like composite model for Language Modeling (proxy).

    Flow: tokens -> CALM encoder (blocks) -> VelmCore -> decoder
    """
    def __init__(self, vocab_size, block_size=4, embed_dim=64, latent_dim=64, state_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.encoder = CALMEncoder(vocab_size, block_size=block_size, embed_dim=embed_dim, latent_dim=latent_dim)
        self.core = VelmCore(latent_dim=latent_dim, state_dim=state_dim)
        self.decoder = CALMDecoder(latent_dim=state_dim, hidden=state_dim, vocab_size=vocab_size, block_size=block_size)

    def forward(self, history, use_adapter=False, return_cib_loss=False):
        # history: (B, L) where L does not need to be divisible by block_size
        B, L = history.shape
        K = self.block_size

        # Adaptive K sequence chunking: allow sequences not divisible by block_size
        latents = []
        k_logits_list = [] if return_cib_loss else None

        start_idx = 0
        while start_idx < L:
            chunk = history[:, start_idx:start_idx+K]
            if return_cib_loss:
                lat, k_logits = self.encoder(chunk, return_k_logits=True)
                k_logits_list.append(k_logits.unsqueeze(1))
            else:
                lat = self.encoder(chunk)
            latents.append(lat.unsqueeze(1))
            start_idx += K

        latents = torch.cat(latents, dim=1)  # (B, n_blocks, latent)

        states = self.core(latents, use_adapter=use_adapter)

        if return_cib_loss:
            # Conditional Information Bottleneck (CIB) heuristic loss
            # Enforce that the latent norm is bounded (pruning cognitive bloat)
            # This is a structural penalty to compress continuous vectors
            cib_loss = latents.norm(p=2, dim=2).mean()
            k_logits_tensor = torch.cat(k_logits_list, dim=1) if k_logits_list else None
            return states, latents, cib_loss, k_logits_tensor

        return states, latents

    def decode_block_state(self, state_block, target_K=None):
        # state_block: (B, state_dim)
        return self.decoder(state_block, target_K=target_K)

    @property
    def adapter(self):
        return self.core.adapter
