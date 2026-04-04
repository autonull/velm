"""VELM inference — JAX/Equinox implementation.

- qttt: Query-only test-time training for long-context adaptation
- cib_budget: CIB-based adaptive reasoning budget controller
"""

from .qttt import extract_query_params, qttt_span_loss, apply_qttt, generate_with_qttt
from .cib_budget import (
    CIBBudgetController,
    estimate_difficulty,
    compute_info_gain,
    should_continue_reasoning,
    allocate_static_budget,
)

__all__ = [
    "extract_query_params",
    "qttt_span_loss",
    "apply_qttt",
    "generate_with_qttt",
    "CIBBudgetController",
    "estimate_difficulty",
    "compute_info_gain",
    "should_continue_reasoning",
    "allocate_static_budget",
]
