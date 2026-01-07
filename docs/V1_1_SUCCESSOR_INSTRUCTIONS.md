# stabilitycheck v1.1: "Perfect" instruction to successor (copy-paste)

## 0) Premise (non-negotiables)

- `stabilitycheck/engine.py` **does computation only**. No plotting, no file I/O.
- `stabilitycheck/runner.py` **writes all paper artifacts** to disk.
- **Do not change** the `EstimationResult` interface/field names:
  - `ok, psi_hat, U, p, Q, n, msg`
- Internal standard:
  - `U` is **covariance** (k×k). If an adapter only has SE vector, it must be promoted with `as_cov(se)`.
- Naming must remain:
  - `S_wc, Neff, K_rule, mu (mu_adv), g_vec, g_scalar`

## 1) [Required] Stage1 split / cross-fit (post-search validity seal)

### Absolute rules

- Learn `mu_adv` **only on the train fold**.
- On the test fold, **mu must stay fixed**.
  - **No re-estimating lambda on test**.
  - **No recomputing mu on test**.

### Test-fold score

For test fold, use:

- `S_wc_test = sum_d mu_d_train * g_d_test`

### Definition of Done (DoD)

When `runner` is executed with `split=True`, the following files MUST be created:

- `paper_outputs/<exp_id>/<tier>/split_summary.csv`
- `paper_outputs/<exp_id>/<tier>/Kcurve_split.csv`
- `paper_outputs/<exp_id>/<tier>/Kcurve_split.png`
- `paper_outputs/<exp_id>/<tier>/weights_train_topshare.csv`

Additionally, the demo run must finish (smoke test) with:

- `python -m stabilitycheck.cli --demo --all --split`

## 2) [Required] K -> lambda root-finding + QC (softmax)

### Absolute rules

- The input knob is **K**, not lambda.
- Lambda is solved ...
