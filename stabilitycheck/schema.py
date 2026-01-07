from __future__ import annotations
from typing import Dict, List, Any
from .configs import DesignSpec

def spec_to_features(spec: DesignSpec) -> Dict[str, float]:
    m = str(spec.get("m",""))
    p = spec.get("p", {}) or {}
    s = spec.get("s", {}) or {}
    e = spec.get("e", {}) or {}
    i = spec.get("i", {}) or {}

    transform = str(p.get("transform", p.get("y_transform","")))
    winsor_p = p.get("winsor_p", None)

    def fbool(x: bool) -> float:
        return 1.0 if bool(x) else 0.0

    feats: Dict[str, float] = {
        "m_is_DID": fbool(m=="DID"),
        "m_is_IV": fbool(m=="IV"),
        "m_is_GLM": fbool(m=="GLM"),
        "drop_last_k": float(s.get("drop_last_k", 0)),
        "uFE": float(e.get("uFE", 0)),
        "tFE": float(e.get("tFE", 0)),
        "control_L": float(e.get("control_L", 0)),
        "transform_log1p": fbool(transform=="log1p"),
        "winsor_p": float(0.0 if winsor_p is None else winsor_p),
        "est_is_OLS": fbool(str(e.get("estimator",""))=="OLS"),
        "est_is_GLM": fbool(str(e.get("estimator",""))=="GLM"),
        "cluster_unit": fbool(str(i.get("cluster",""))=="unit"),
    }
    return feats

def feature_columns(feats_list: List[Dict[str,float]]) -> List[str]:
    return sorted({k for f in feats_list for k in f.keys()})

def design_key(spec: DesignSpec) -> str:
    import json
    return json.dumps(spec, sort_keys=True)


def make_designs_for_tier(schema: object, tier: str) -> List[DesignSpec]:
    """Return the list of DesignSpec objects for a given tier.

    Runner/engine code may pass either:
      - configs.SchemaConfig (preferred)
      - a plain dict-like object with a .tiers mapping

    This helper keeps the public runner API stable while allowing multiple
    schema representations internally.
    """

    if not hasattr(schema, "tiers"):
        raise TypeError("schema must have a .tiers attribute")
    tiers = getattr(schema, "tiers")
    if not isinstance(tiers, dict) or tier not in tiers:
        raise KeyError(f"tier not found in schema.tiers: {tier}")

    tier_obj = tiers[tier]
    if isinstance(tier_obj, list):
        return list(tier_obj)
    if hasattr(tier_obj, "designs"):
        return list(getattr(tier_obj, "designs"))
    raise TypeError(f"schema.tiers['{tier}'] is not a list and has no .designs")
