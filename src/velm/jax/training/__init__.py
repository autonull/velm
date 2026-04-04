"""VELM training — JAX/Equinox implementation.

- eggroll: Low-rank evolution strategy optimizer
- fitness: Multi-objective fitness (quality + compression)
"""

from .eggroll import (
    generate_low_rank_perturbation,
    perturb_pytree,
    EGGROLLState,
    create_eggroll_optimizer,
    eggroll_step,
    discretize_update_int8,
)
from .fitness import quality_fitness, compression_penalty, combined_fitness

__all__ = [
    "generate_low_rank_perturbation",
    "perturb_pytree",
    "EGGROLLState",
    "create_eggroll_optimizer",
    "eggroll_step",
    "discretize_update_int8",
    "quality_fitness",
    "compression_penalty",
    "combined_fitness",
]
