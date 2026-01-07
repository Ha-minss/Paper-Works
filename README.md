# stabilitycheck (universal) — v1.1.0

`stabilitycheck` is a **universal robustness checker** for empirical research.
You give it:

- a **design space** (many reasonable specifications)
- one or more **anchor** specifications (your “main” and key alternatives)
- an **adapter** (how to estimate a spec)

and it outputs **stability scores**, **weighting diagnostics**, and **paper-ready artifacts** (tables/plots) that help you defend your identification choices.

This repository is intentionally **adapter-driven**: the *engine is model-agnostic*.

---

## What’s included in v1.1.0

### v1.1 handoff docs (GitHub-ready)

- Successor instruction (copy-paste): `docs/V1_1_PERFECT_INSTRUCTION.md`
- Paper output checklist (Definition of Done): `docs/V1_1_PAPER_OUTPUTS_CHECKLIST.md`

### 1) Core engine
- `stabilitycheck/engine.py`: computes design table `D`, anchor comparisons, `g`-vectors, and Stage-1 worst-case score `S_wc`.
- Supports **vector ψ** (multiple parameters of interest) via `psi_index`.
- Supports `U` as either **SE-vector** or **covariance matrix** (internally standardized to cov).

### 2) Full pipeline runner
- `stabilitycheck/runner.py`: one-call paper pipeline
  - Tier1/2/3 batch run
  - K-curve / w-curve
  - Stage-1 null (permutation) testification
  - Stage-2 block/rolling reproducibility
  - Stage-2.5 RF explanation (bootstrap CI + PDP/ICE)

### 3) Built-in adapters (real implementations)
All adapters follow the same contract:
`fit(df, spec) -> EstimationResult(ok, psi_hat, U, p, Q, n, msg)`

- `DIDAdapter` — panel OLS DID with FE + cluster SE
- `EventStudyAdapter` — panel event-study (lead/lag dummies) with FE + cluster SE
- `GLMLogAdapter` — GLM with log-link (PPML-style). Good default for nonnegative outcomes.
- `OLSAdapter` — plain OLS (+ optional FE demeaning)
- `IV2SLSAdapter` — 2SLS / IV with optional FE demeaning + cluster SE
- `RDAdapter` — sharp RD (WLS local polynomial) + optional fuzzy RD (2SLS local)
- `DMLPLRAdapter` — **real DML** (PLR, cross-fitting) with sklearn learners + cluster option
- `BayesOLSAdapter` — conjugate Bayesian OLS (Normal–Inverse-Gamma), returns posterior for ψ
- `StructuralDDCAdapter` — **dynamic discrete choice** (binary logit shocks) via NFXP + VFI
  - Includes an internal solver in `stabilitycheck/structural/ddc.py`

### 4) Large-|D| sampling utilities
- `stabilitycheck/sampling.py`: space-filling sampling + refinement helpers

### 5) Tests + CI scaffolding
- `tests/` smoke tests
- `.github/workflows/tests.yml` (pytest)

---

## Quick start

```python
import numpy as np
import pandas as pd

from stabilitycheck.adapters import default_registry
from stabilitycheck.configs import EngineConfig
from stabilitycheck.runner import run_all

# 1) prepare your dataframe df
# df must match the adapter’s colmap (see adapter docstrings)

registry = default_registry()

cfg = EngineConfig(
  outdir="paper_outputs",
  seed=7,
  alpha_g=0.10,
)

# 2) create your design space (list of `spec` dicts)
# for DID/event-study, see stabilitycheck/demo.py for an example

designs = [...]  # many specs

# 3) run pipeline
run_all(
  df=df,
  cfg=cfg,
  designs=designs,
  registry=registry,
  main_anchor=designs[0],
  other_anchors=[designs[1], designs[2]],
)
```

---

## How to publish this on GitHub

Yes — you can upload this folder as a normal repo.
Recommended structure:

- keep `stabilitycheck/` as the package
- keep `pyproject.toml` at repo root
- keep `tests/` and `.github/workflows/` for CI

Then users can install with:

```bash
pip install -e .
```

---

## Notes on “Structural/Bayesian” scope

This version **does include** a real structural solver (binary DDC with logit shocks) to satisfy “internal structural estimation.”
If you later need a very specific model (e.g., nested logit, continuous state, CCP estimators, multiple agents), the intended workflow is:

1) keep the structural solver in `stabilitycheck/structural/`
2) create a new adapter that calls it
3) register it in `default_registry()`

---

## License
MIT
