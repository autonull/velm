import torch
import torch.nn as nn
from .calm import CALMEncoder, CALMDecoder
from .miras import MirasMemory
from .swa import SWALayer, RMSNorm
from .qttt import Adapter

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

    def forward(self, history, use_adapter=False):
        # history: (B, L) where L divisible by block_size
        B, L = history.shape
        K = self.block_size
        assert L % K == 0
        n_blocks = L // K
        blocks = history.view(B, n_blocks, K)
        # encode each block
        latents = []
        for i in range(n_blocks):
            lat = self.encoder(blocks[:, i, :])
            latents.append(lat.unsqueeze(1))
        latents = torch.cat(latents, dim=1)  # (B, n_blocks, latent)

        states = self.core(latents, use_adapter=use_adapter)
        return states, latents

    def decode_block_state(self, state_block):
        # state_block: (B, state_dim)
        return self.decoder(state_block)

    @torch.no_grad()
    def generate(self, prompt, max_new_blocks=10, use_adapter=False, cib_threshold=0.01, min_reasoning_blocks=2):
        """
        Generates new tokens using dynamic CIB inference.
        Reasoning is terminated early if the information gain (cosine distance)
        between consecutive latent states falls below cib_threshold.
        """
        self.eval()
        B, L = prompt.shape
        device = prompt.device

        # Ensure prompt is divisible by block size
        K = self.block_size
        if L % K != 0:
            pad_len = K - (L % K)
            # Pad with 0s for simplicity
            prompt = torch.cat([prompt, torch.zeros(B, pad_len, dtype=torch.long, device=device)], dim=1)

        current_history = prompt
        generated_blocks = []

        prev_state = None
        for step in range(max_new_blocks):
            states, latents = self(current_history, use_adapter=use_adapter)

            # The state corresponding to the last block
            last_state = states[:, -1, :] # (B, state_dim)

            # CIB Dynamic Termination Check
            if prev_state is not None and step >= min_reasoning_blocks:
                # Calculate cosine distance as information gain proxy
                # High similarity -> low distance -> low information gain
                cos_sim = torch.nn.functional.cosine_similarity(prev_state, last_state, dim=-1)
                info_gain = 1.0 - cos_sim

                # If info_gain across the batch is generally below threshold, terminate early
                if info_gain.mean().item() < cib_threshold:
                    break

            prev_state = last_state

            # Decode the last state into new tokens
            new_logits = self.decode_block_state(last_state) # (B, K, vocab_size)
            new_tokens = new_logits.argmax(dim=-1) # (B, K)

            generated_blocks.append(new_tokens)
            current_history = torch.cat([current_history, new_tokens], dim=1)

        self.train()
        if not generated_blocks:
            return prompt
        return torch.cat(generated_blocks, dim=1)

    @property
    def adapter(self):
        return self.core.adapter
