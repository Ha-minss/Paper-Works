"""Structural estimation helpers.

This subpackage contains *fully self-contained* (pure Python) solvers
that can be used by adapters.

Currently included:
- ddc: Dynamic Discrete Choice (single-agent, logit shocks) via NFXP.
"""

from .ddc import (
    estimate_binary_logit_ddc,
    simulate_binary_logit_ddc,
)

__all__ = ["estimate_binary_logit_ddc", "simulate_binary_logit_ddc"]
