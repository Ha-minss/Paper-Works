import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def test_import_and_registry():
    import stabilitycheck
    from stabilitycheck.adapters import default_registry

    reg = default_registry()
    assert "did" in reg.families()
    assert "rd" in reg.families()
    assert "structural_ddc" in reg.families()


def test_rd_runs():
    from stabilitycheck.adapters import default_registry

    np.random.seed(0)
    n = 1500
    x = np.random.uniform(-1, 1, size=n)
    y = 1.0 + 2.0 * (x >= 0) + 0.3 * x + np.random.normal(scale=1.0, size=n)
    df = pd.DataFrame({"x": x, "y": y})

    reg = default_registry()
    rd = reg.get("rd")()

    spec = {
        "m": "rd",
        "p": {"y_transform": "level"},
        "s": {"h": 0.6},
        "e": {"y": "y", "x": "x", "cutoff": 0.0, "order": 1, "kernel": "triangular"},
        "i": {"psi_index": 0},
    }
    res = rd.fit(df, spec)
    assert res.ok
    assert res.psi_hat.shape == (1,)
    assert res.U.shape == (1, 1)


def test_structural_ddc_runs():
    from stabilitycheck.structural.ddc import simulate_binary_logit_ddc
    from stabilitycheck.adapters import default_registry

    df = simulate_binary_logit_ddc(n_ids=200, t_periods=12, beta=0.9, theta=(1.2, -0.5), seed=0)

    reg = default_registry()
    ad = reg.get("structural_ddc")()

    spec = {
        "m": "structural_ddc",
        "e": {
            "id": "id",
            "t": "t",
            "state": "state",
            "choice": "choice",
            "beta": 0.9,
            "max_iter": 500,
            "tol": 1e-10,
            "start": [0.5, -0.1],
        },
        "i": {"psi_index": 0},
    }

    res = ad.fit(df, spec)
    assert res.ok
    assert res.psi_hat.shape == (2,)
    assert res.U.shape == (2, 2)
