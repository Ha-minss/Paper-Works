# paper_outputs/ Artifact Checklist (v1.1 DoD)

**Folder convention**

- Root: `paper_outputs/<exp_id>/`
- Per tier: `paper_outputs/<exp_id>/<tier_name>/`

`exp_id` should be stable and reproducible given (`seed`, `dataset_label`, `tier set`).

---

## A) Common (always) — per tier (8)

Inside `paper_outputs/<exp_id>/<tier_name>/`:

1. `design_table.csv`
   - Minimum columns:
     - `design_id, spec_key, family, ok, n, psi_hat_json, U_json, p_json, g_vec_json, g_scalar, msg, Q_json`
     - plus all knob feature columns returned by `schema.spec_to_features(...)`

2. `tier_summary.csv`
   - One row summary:
     - `tier, n_total, n_ok, n_fail, K_rule, S_wc_mainK, max_g, Neff_mainK, lambda_mainK`

3. `Kcurve.csv`
   - Columns:
     - `K, lambda, Neff, S_wc, max_g, n_ok_designs, converged, rel_err`

4. `Kcurve.png`

5. `wcurve.csv` (if w-curve is enabled)
   - Columns:
     - `w2, w3, S_wc, max_g`

6. `wcurve.png` (if w-curve is enabled)

7. `softmax_qc.csv`
   - Columns:
     - `K, lambda, Neff, rel_err, iters, converged, bracket_lo, bracket_hi`

8. (Optional but recommended) `scatter_tau_mu.png`, `weights_cdf.png`

---

## B) Split/Cross-fit Stage1 — when `split=True` (4)

Inside `paper_outputs/<exp_id>/<tier_name>/`:

9. `split_summary.csv`
   - Columns:
     - `fold, n_train, n_test, K, lambda_train, Neff_train, S_wc_test, max_g_test`

10. `Kcurve_split.csv`
    - Columns:
      - `K, S_wc_split_mean, S_wc_split_se`

11. `Kcurve_split.png`

12. `weights_train_topshare.csv`
    - Columns:
      - `fold, top_frac, weight_share`

---

## C) Large-|D| — when `largeD=True` (4)

Inside `paper_outputs/<exp_id>/<tier_name>/`:

13. `LargeD_sampling_plan.json`

14. `LargeD_tightening.csv`
    - Columns:
      - `iter, stage, n_designs, S_wc, max_g, coverage_p50, coverage_p90`

15. `LargeD_tightening.png`

16. `coverage.json`

---

## D) Stage1 Testification — when `testify=True` (3)

Inside `paper_outputs/<exp_id>/`:

17. `Stage1_test_report.csv`
    - Columns:
      - `tier, K, S_obs, critical_value, p_value_mc, alpha, B_used, decision, null_mode`

18. `Stage1_null_hist.png`

19. `size_power_table.csv`
    - Minimum 3 rows:
      - `StableNull, LocalViolation, RegimeShift`

---

## E) Experiment root metadata (2)

Inside `paper_outputs/<exp_id>/`:

20. `meta_run.json`

21. `config_snapshot.json`

---

## Final DoD

A single execution of runner must create the above artifacts with **exact filenames** and **schema-consistent columns**.
