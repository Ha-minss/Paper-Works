from __future__ import annotations

from .base import BaseAdapter, EstimationResult
from .dispatch import AdapterRegistry
from .did import DIDAdapter
from .glm_log import GLMLogLinkAdapter
from .rd import RDAdapter
from .event_study import EventStudyAdapter
from .dml_plr import DMLPLRAdapter
from .iv_2sls import IV2SLSAdapter
from .bayes_ols import BayesOLSAdapter
from .structural_ddc import StructuralDDCAdapter

from ..configs import SchemaConfig, default_schema_did


__all__ = [
    "BaseAdapter",
    "EstimationResult",
    "AdapterRegistry",
    "default_registry",
    # adapters
    "DIDAdapter",
    "GLMLogLinkAdapter",
    "RDAdapter",
    "EventStudyAdapter",
    "DMLPLRAdapter",
    "IV2SLSAdapter",
    "GMMAdapter",
    "BayesOLSAdapter",
    "StructuralDDCAdapter",
]


def default_registry(schema: SchemaConfig | None = None) -> AdapterRegistry:
    """Return a registry with all built-in adapters registered.

    Parameters
    ----------
    schema:
        Optional SchemaConfig. Some adapters (e.g., DID) use schema defaults such
        as `controls_ranked`. If omitted, `default_schema_did()` is used.

    Notes
    -----
    The registry is case-insensitive on family names.
    """

    if schema is None:
        schema = default_schema_did()

    reg = AdapterRegistry()

    # Reduced-form DID / panel OLS with FE demeaning.
    reg.register("did", lambda: DIDAdapter(schema))

    # GLM with log link (Poisson family by default). Useful as PPML-style baseline.
    reg.register("glm_log", lambda: GLMLogLinkAdapter())

    # RD/RDD (local polynomial with bandwidth/grid knobs).
    reg.register("rd", lambda: RDAdapter())

    # Event-study (dynamic effects around an event time).
    reg.register("event_study", lambda: EventStudyAdapter())

    # Double/Debiased ML for partially linear regression.
    reg.register("dml_plr", lambda: DMLPLRAdapter())

    # Panel-IV (2SLS with fixed effects options).
    reg.register("panel_iv_2sls", lambda: IV2SLSAdapter())

    # Generic GMM wrapper (moment-based estimators).

    # Bayesian linear regression (conjugate normal-inverse-gamma).
    reg.register("bayes_linear", lambda: BayesOLSAdapter())

    # Structural: dynamic discrete choice (toy solver wrapper).
    reg.register("structural_ddc", lambda: StructuralDDCAdapter())

    # Backward-compat alias
    reg.register("ddc", lambda: StructuralDDCAdapter())

    return reg
