from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any
import numpy as np

DesignSpec = Dict[str, Any]

@dataclass
class TierConfig:
    name: str
    designs: List[DesignSpec]

@dataclass
class EngineConfig:
    outdir: str = "paper_outputs"
    # Bookkeeping used by runner (kept here so a single config object can drive
    # both engine + runner without changing the public API).
    exp_id: Optional[str] = None
    dataset: str = "demo"
    tier_names: Sequence[str] = ("Tier1_Core", "Tier2_Contested", "Tier3_WiderStress")
    seed: int = 0

    anchor_main: Optional[DesignSpec] = None
    anchor_cons: Optional[DesignSpec] = None
    anchor_alt:  Optional[DesignSpec] = None
    anchor_agg: str = "max"  # max/mean/min

    alpha_g: float = 0.10
    p_norms: Sequence[Any] = (1, 2, np.inf)
    w2_grid: Sequence[float] = (1.0, 2.0, 3.0)
    w3_grid: Sequence[float] = (0.0, 0.25, 0.5)

    # Defaults for a single scalarization used by some runner routines
    # (when not sweeping grids).
    pnorm: Any = 2
    w2: float = 2.0
    w3: float = 0.5

    k_min: int = 10
    rho: float = 0.05
    k_grid_base: Sequence[int] = (1,2,5,10,20,50,100,200,500)

    do_stage1_test: bool = True
    null_B: int = 200
    alpha_test: float = 0.05
    null_mode: str = "permute_treat"

    # Stage1 testification: optional size/power sanity checks
    size_power_B: int = 200
    size_power_R: int = 3

    do_stage2: bool = True
    block_splits: int = 3

    do_stage25: bool = True
    rf_trees: int = 600
    rf_boot: int = 500
    pdp_grid: int = 20
    ice_n: int = 200

    do_largeD_sampling: bool = False
    largeD_threshold: int = 2000
    largeD_method: str = "lhs"  # "lhs" or "sobol"
    largeD_M0: int = 256
    largeD_M1: int = 128
    largeD_top_frac: float = 0.10
    largeD_sweep: int = 128
    largeD_spike_sweep: bool = True
    # Backward-compat (deprecated; not used by runner)
    largeD_refine_rounds: int = 2
    largeD_topq: float = 0.10

@dataclass
class SchemaConfig:
    controls_ranked: List[str] = field(default_factory=list)
    tiers: Dict[str, TierConfig] = field(default_factory=dict)

def _did_spec(uFE: int, tFE: int, drop_last_k: int, ctrl_L: int, transform: str, winsor_p: Optional[float],
             estimator: str = "OLS") -> DesignSpec:
    return {
        "m": "DID",
        "p": {"transform": transform, "winsor_p": winsor_p},
        "s": {"drop_last_k": int(drop_last_k)},
        "e": {"estimator": estimator, "uFE": int(uFE), "tFE": int(tFE), "control_L": int(ctrl_L)},
        "i": {"cluster": "unit", "psi_index": 0},
    }

def default_schema_did(controls_ranked: List[str] | None = None) -> SchemaConfig:
    if controls_ranked is None:
        controls_ranked = ["x1","x2","x3"]

    def tier(name: str, fe_opts, drop_last_k, control_levels, transforms, winsor_ps) -> TierConfig:
        designs: List[DesignSpec] = []
        for (uFE,tFE) in fe_opts:
            for dk in drop_last_k:
                for cl in control_levels:
                    for tr in transforms:
                        for wp in winsor_ps:
                            designs.append(_did_spec(uFE,tFE,dk,cl,tr,wp,estimator="OLS"))
        return TierConfig(name=name, designs=designs)

    tiers = {
        "Tier1_Core": tier(
            "Tier1_Core",
            fe_opts=[(1,1)],
            drop_last_k=[0],
            control_levels=[1,2],
            transforms=["log1p"],
            winsor_ps=[None, 0.01],
        ),
        "Tier2_Contested": tier(
            "Tier2_Contested",
            fe_opts=[(1,1),(1,0),(0,1)],
            drop_last_k=[0,3,6],
            control_levels=[0,1,2,3],
            transforms=["level","log1p"],
            winsor_ps=[None, 0.01],
        ),
        "Tier3_WiderStress": tier(
            "Tier3_WiderStress",
            fe_opts=[(1,1),(1,0),(0,1),(0,0)],
            drop_last_k=[0,3,6,9,12],
            control_levels=[0,1,2,3],
            transforms=["level","log1p"],
            winsor_ps=[None, 0.01, 0.05],
        ),
    }
    return SchemaConfig(controls_ranked=list(controls_ranked), tiers=tiers)
