"""VELM model architecture — JAX/Equinox implementation.

Provides the canonical research-scale model components:
  - CONFIGS, QWEN35_VOCAB_SIZE: model configurations
  - CALMAutoencoder: token ↔ latent compression
  - VELMBackbone: Miras + SWA hybrid sequence processor
  - EnergyHead: energy-based generative head
  - VELM: complete model composing all components
"""

from .config import CONFIGS, QWEN35_VOCAB_SIZE
from .autoencoder import CALMAutoencoder, batch_ae_loss, reconstruction_accuracy
from .miras_backbone import VELMBackbone, MirasMemoryLayer, SlidingWindowAttention
from .energy_head import EnergyHead, energy_score, energy_loss
from .velm import VELM

__all__ = [
    "CONFIGS",
    "QWEN35_VOCAB_SIZE",
    "CALMAutoencoder",
    "batch_ae_loss",
    "reconstruction_accuracy",
    "VELMBackbone",
    "MirasMemoryLayer",
    "SlidingWindowAttention",
    "EnergyHead",
    "energy_score",
    "energy_loss",
    "VELM",
]
