from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .adapters import default_registry
from .assumptions import attach_A_columns, compute_delta_curve, default_delta_grid_for_family, plot_assumption_curve_to
from .configs import EngineConfig, default_schema_did
from .contract import validate_design_table
from .dgp import simulate_panel
from .engine import compute_anchors, compute_design_table, compute_g_scalar, make_k_grid
from .plotting import save_placeholder_png
from .report import ensure_dir, save_csv_file, save_json_file
from .sampling import largeD_tighten
from .softmax import stage1_score
from .testification import mc_stage1_test, size_and_power_table


def _tb1(ex: BaseException) -> str:
    s = "".join(traceback.format_exception_only(type(ex), ex)).strip()
    return s.splitlines()[0] if s else repr(ex)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stabilitycheck.runner")

    p.add_argument("--demo", action="store_true", help="Run on built-in demo data")
    p.add_argument("--all", action="store_true", help="Run all tiers")
    p.add_argument("--split", action="store_true", help="Run split/crossfit stage1")
    p.add_argument("--largeD", action="store_true", help="Run large-D tightening (skips if |D| is small)")
    p.add_argument("--testify", action="store_true", help="Run testification (null + size/power)")
    p.add_argument("--assumption", action="store_true", help="Compute δ-knob assumption curves")

    # Speed knobs (defaults intentionally small for quick sanity runs)
    p.add_argument("--null_B", type=int, default=40, help="MC null draws for Stage1 test")
    p.add_argument("--size_power_B", type=int, default=40, help="MC draws per scenario for size/power")
    p.add_argument("--size_power_R", type=int, default=6, help="Number of scenarios for size/power")

    p.add_argument("--out", type=str, default="paper_outputs", help="Output root directory")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args(argv)


def _placeholder_bundle(tier: str, reason: str) -> Dict[str, Any]:
    empty = pd.DataFrame(
        [
            {
                "tier": tier,
                "K": 1,
                "S_obs": np.nan,
                "critical_value": np.nan,
                "p_value_mc": np.nan,
                "alpha": 0.05,
                "B_used": 0,
                "decision": "SKIP",
                "reason": reason,
            }
        ]
    )
    return {
        "stage1_test_report_df": empty,
        "stage1_null_scores": np.asarray([0.0]),
        "softmax_qc_df": pd.DataFrame(columns=["K", "lambda", "Neff", "converged", "rel_err", "iters", "max_g"]),
        "size_power_table_df": pd.DataFrame(columns=["scenario", "alpha", "size", "power"]),
    }


def save_testification_bundle(bundle: Dict[str, Any], exp_root: str, tier_dir: str) -> None:
    """Save required paper artifacts to BOTH exp_root and tier_dir.

    Even if a stage is skipped, placeholder files are created.
    """

    ensure_dir(exp_root)
    ensure_dir(tier_dir)

    # Stage1 report
    df_report = bundle.get("stage1_test_report_df")
    if df_report is None or not isinstance(df_report, pd.DataFrame):
        df_report = pd.DataFrame()
    save_csv_file(os.path.join(exp_root, "Stage1_test_report.csv"), df_report)
    save_csv_file(os.path.join(tier_dir, "Stage1_test_report.csv"), df_report)

    # Stage1 null hist
    null_scores = bundle.get("stage1_null_scores")
    if null_scores is None:
        save_placeholder_png(os.path.join(exp_root, "Stage1_null_hist.png"), title="null hist (missing)")
        save_placeholder_png(os.path.join(tier_dir, "Stage1_null_hist.png"), title="null hist (missing)")
    else:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = np.asarray(null_scores, dtype=float).reshape(-1)
            fig = plt.figure()
            plt.hist(xs[np.isfinite(xs)], bins=20)
            plt.title("Stage1 null histogram")
            fig.savefig(os.path.join(exp_root, "Stage1_null_hist.png"), dpi=140, bbox_inches="tight")
            fig.savefig(os.path.join(tier_dir, "Stage1_null_hist.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            save_placeholder_png(os.path.join(exp_root, "Stage1_null_hist.png"), title="null hist (error)")
            save_placeholder_png(os.path.join(tier_dir, "Stage1_null_hist.png"), title="null hist (error)")

    # size/power table
    df_sp = bundle.get("size_power_table_df")
    if df_sp is None or not isinstance(df_sp, pd.DataFrame):
        df_sp = pd.DataFrame()
    save_csv_file(os.path.join(exp_root, "size_power_table.csv"), df_sp)
    save_csv_file(os.path.join(tier_dir, "size_power_table.csv"), df_sp)

    # softmax qc
    df_qc = bundle.get("softmax_qc_df")
    if df_qc is None or not isinstance(df_qc, pd.DataFrame):
        df_qc = pd.DataFrame()
    save_csv_file(os.path.join(exp_root, "softmax_qc.csv"), df_qc)
    save_csv_file(os.path.join(tier_dir, "softmax_qc.csv"), df_qc)


def _split_crossfit_stage1(
    cfg: EngineConfig,
    schema: Any,
    registry: Any,
    df: pd.DataFrame,
    tier: str,
    anchors: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    id_col = "unit" if "unit" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        return {"error": "no unit/id column for split"}
    ids = np.unique(df[id_col].to_numpy())
    rng.shuffle(ids)
    mid = len(ids) // 2
    A = set(ids[:mid])

    df_A = df[df[id_col].isin(A)].copy()
    df_B = df[~df[id_col].isin(A)].copy()

    out = {}
    for tag, dsub in [("A", df_A), ("B", df_B)]:
        Dt = compute_design_table(cfg, schema, registry, df=dsub, tier_name=tier)
        g = compute_g_scalar(cfg, Dt, anchors)
        g_ok = g[np.isfinite(g)]
        if g_ok.size == 0:
            out[tag] = {"n": 0, "K_rule": 1, "S_wc": np.nan}
        else:
            K_rule = min(int(g_ok.size), max(cfg.k_min, int(np.ceil(cfg.rho * g_ok.size))))
            out[tag] = {"n": int(g_ok.size), "K_rule": int(K_rule), **stage1_score(g_ok, K_rule)}
    return out


def _maybe_largeD(cfg: EngineConfig, tier_designs: List[dict]) -> Dict[str, Any]:
    D = list(tier_designs)
    if not bool(getattr(cfg, "do_largeD_sampling", False)):
        return {
            "sampling_plan": {"threshold": cfg.largeD_threshold, "note": "largeD disabled"},
            "tightening_df": pd.DataFrame(),
            "coverage": {},
            "skipped": True,
        }
    if len(D) <= int(cfg.largeD_threshold):
        return {
            "sampling_plan": {"threshold": cfg.largeD_threshold, "note": "|D| below threshold"},
            "tightening_df": pd.DataFrame(),
            "coverage": {},
            "skipped": True,
        }

    try:
        td = largeD_tighten(
            D,
            seed=int(cfg.seed),
            method=str(cfg.largeD_method),
            M0=int(cfg.largeD_M0),
            M1=int(cfg.largeD_M1),
            top_frac=float(cfg.largeD_top_frac),
            M_sweep=int(cfg.largeD_sweep),
        )
        td["skipped"] = False
        return td
    except Exception as ex:
        return {
            "sampling_plan": {"threshold": cfg.largeD_threshold, "note": f"largeD error: {_tb1(ex)}"},
            "tightening_df": pd.DataFrame(),
            "coverage": {},
            "skipped": True,
        }


def run_paper_pipeline(args: argparse.Namespace) -> str:
    np.random.seed(args.seed)

    registry = default_registry()
    schema = default_schema_did()

    if args.demo:
        # Intentionally smaller than the paper defaults so the DoD runs fast.
        df = simulate_panel(seed=args.seed, N_units=30, T_periods=36)
        dataset = "demo"
    else:
        raise RuntimeError("Non-demo datasets are not wired in this build.")

    exp_id = f"{dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_root = os.path.join(args.out, exp_id)
    ensure_dir(exp_root)

    tiers = list(schema.tiers.keys()) if args.all else ["Tier1_Core"]

    for tier in tiers:
        tier_dir = os.path.join(exp_root, tier)
        ensure_dir(tier_dir)

        tier_obj = schema.tiers[tier]
        tier_designs = tier_obj.designs

        cfg = EngineConfig(dataset=dataset, exp_id=exp_id, seed=args.seed)
        cfg = replace(cfg, anchor_main=tier_designs[0] if len(tier_designs) > 0 else None)
        cfg = replace(cfg, anchor_cons=tier_designs[1] if len(tier_designs) > 1 else None)
        cfg = replace(cfg, anchor_alt=tier_designs[2] if len(tier_designs) > 2 else None)
        cfg = replace(cfg, null_B=int(args.null_B), size_power_B=int(args.size_power_B), size_power_R=int(args.size_power_R))
        cfg = replace(cfg, do_largeD_sampling=bool(args.largeD))

        # 1) Design table (fail-soft)
        try:
            D = compute_design_table(cfg, schema, registry, df=df, tier_name=tier)
        except Exception as ex:
            D = validate_design_table(pd.DataFrame())
            save_json_file(os.path.join(tier_dir, "errors.json"), {"design_table": _tb1(ex)})

        save_csv_file(os.path.join(tier_dir, "design_table.csv"), D)

        # 2) Anchors (fail-soft)
        anchors = compute_anchors(cfg, df=df, adapter_registry=registry, schema=schema, tier_name=tier)
        save_json_file(os.path.join(tier_dir, "anchors.json"), anchors)

        # 3) Stage1 g + curve
        g = compute_g_scalar(cfg, D, anchors)
        g_ok = g[np.isfinite(g)]

        if g_ok.size > 0:
            K_grid = make_k_grid(cfg, int(g_ok.size))
            stage1_df = pd.DataFrame([{"tier": tier, "K": int(K), **stage1_score(g_ok, int(K))} for K in K_grid])
        else:
            K_grid = [1]
            stage1_df = pd.DataFrame([{"tier": tier, "K": 1, "S_wc": np.nan}])

        save_csv_file(os.path.join(tier_dir, "stage1_curve.csv"), stage1_df)

        # 4) Split/crossfit
        if args.split:
            try:
                split = _split_crossfit_stage1(cfg, schema, registry, df, tier, anchors, seed=args.seed)
            except Exception as ex:
                split = {"error": _tb1(ex)}
            save_json_file(os.path.join(tier_dir, "split_stage1.json"), split)

        # 5) largeD tightening
        if args.largeD:
            largeD_res = _maybe_largeD(cfg, tier_designs)
            save_json_file(os.path.join(tier_dir, "largeD.json"), {
                "sampling_plan": largeD_res.get("sampling_plan", {}),
                "coverage": largeD_res.get("coverage", {}),
                "skipped": bool(largeD_res.get("skipped", False)),
            })
            save_csv_file(os.path.join(tier_dir, "largeD_tightening.csv"), largeD_res.get("tightening_df", pd.DataFrame()))

        # 6) δ-knob assumption curve
        if args.assumption:
            try:
                Dt = attach_A_columns(D)
                g_full = compute_g_scalar(cfg, Dt, anchors)
                fam = str(Dt["family"].iloc[0]) if len(Dt) else "did"
                delta_grid = default_delta_grid_for_family(fam)
                K_grid_A = make_k_grid(cfg, int(np.sum(np.isfinite(g_full))))
                A_name = str(Dt["A_name"].iloc[0]) if (len(Dt) and "A_name" in Dt.columns) else "A"
                curve = compute_delta_curve(
                    Dt,
                    g_full,
                    K_grid=K_grid_A,
                    delta_grid=delta_grid,
                    family=fam,
                    tier=tier,
                    A_name=A_name,
                )
                save_csv_file(os.path.join(tier_dir, "AssumptionCurve.csv"), curve)
                K_plot = int(K_grid_A[-1]) if len(K_grid_A) else 1
                plot_assumption_curve_to(os.path.join(tier_dir, "AssumptionCurve.png"), curve, K=K_plot)
            except Exception as ex:
                save_csv_file(os.path.join(tier_dir, "AssumptionCurve.csv"), pd.DataFrame())
                save_placeholder_png(os.path.join(tier_dir, "AssumptionCurve.png"), title=f"assumption curve error: {_tb1(ex)}")

        # 7) Testification bundle (always save placeholders)
        if args.testify:
            try:
                if g_ok.size >= 3:
                    # stage1 test at rule-K
                    K_rule = min(int(g_ok.size), max(cfg.k_min, int(np.ceil(cfg.rho * g_ok.size))))
                    test = mc_stage1_test(g_ok, K_rule, alpha=0.05, B=cfg.null_B, rng=np.random.default_rng(cfg.seed))

                    # softmax QC grid
                    qc_rows = []
                    for K in K_grid:
                        # mc_stage1_test already calls solve_lambda; but we store minimal
                        qc_rows.append({"K": int(K), "lambda": float(test.get("lambda", np.nan)), "Neff": float(test.get("Neff", np.nan)), "converged": bool(test.get("converged", False)), "rel_err": float(test.get("rel_err", np.nan)), "iters": int(test.get("iters", 0)), "max_g": float(np.nanmax(g_ok))})
                    softmax_qc_df = pd.DataFrame(qc_rows)

                    sp = size_and_power_table(
                        g_ok,
                        K_rule,
                        alpha=0.05,
                        B=cfg.size_power_B,
                        R=cfg.size_power_R,
                        rng=np.random.default_rng(cfg.seed + 13),
                    )

                    bundle = {
                        "stage1_test_report_df": pd.DataFrame([
                            {
                                "tier": tier,
                                "K": int(K_rule),
                                "S_obs": float(test.get("S_obs", np.nan)),
                                "critical_value": float(test.get("critical_value", np.nan)),
                                "p_value_mc": float(test.get("p_value_mc", np.nan)),
                                "alpha": float(test.get("alpha", 0.05)),
                                "B_used": int(test.get("B_used", cfg.null_B)),
                                "decision": str(test.get("decision", "NA")),
                                "null_mode": str(test.get("null_mode", "mc")),
                            }
                        ]),
                        "stage1_null_scores": test.get("null_scores", np.asarray([0.0])),
                        "softmax_qc_df": softmax_qc_df,
                        "size_power_table_df": sp,
                    }
                else:
                    bundle = _placeholder_bundle(tier, reason="too few finite g")
            except Exception as ex:
                bundle = _placeholder_bundle(tier, reason=f"testify error: {_tb1(ex)}")

            save_testification_bundle(bundle, exp_root, tier_dir)

    return exp_root


def main(argv: List[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    exp_root = run_paper_pipeline(args)
    print(exp_root)


if __name__ == "__main__":
    main()
