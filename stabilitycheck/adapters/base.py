from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class EstimationResult:
    ok: bool
    psi_hat: np.ndarray
    U: np.ndarray
    p: Optional[np.ndarray] = None
    Q: Dict[str, Any] = field(default_factory=dict)
    n: int = 0
    msg: str = ""

class BaseAdapter:
    family: str = "BASE"
    def fit(self, data, design_spec) -> EstimationResult:
        raise NotImplementedError

    def nullify(self, data, mode: str, rng):
        return None
