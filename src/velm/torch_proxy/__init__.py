"""PyTorch proxy implementations for VELM components.

This subpackage provides lightweight, well-documented, easy-to-run
PyTorch versions of key VELM components intended for fast experiments
and visualization. For research-scale training and evaluation prefer
the JAX/Equinox implementation under `velm.jax.*`.
"""

from .calm import CALMEncoder, CALMDecoder
from .miras import MirasMemory
from .cib import CIBLoss
from .qttt import Adapter
from .eggroll_stub import Eggroll
from .model import VelmFull

__all__ = [
    "CALMEncoder",
    "CALMDecoder",
    "MirasMemory",
    "CIBLoss",
    "Adapter",
    "Eggroll",
    "VelmFull",
]
