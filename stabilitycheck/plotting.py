from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def plot_tau_mu(outdir: str, prefix: str, tau: np.ndarray, mu_panels: dict, g: np.ndarray):
    ensure_outdir(outdir)
    Ks = list(mu_panels.keys())
    fig, axes = plt.subplots(1, len(Ks), figsize=(4.6*len(Ks), 3.4), sharey=True)
    if len(Ks) == 1:
        axes = [axes]
    for ax, K in zip(axes, Ks):
        mu = mu_panels[K]
        sc = ax.scatter(tau, mu, c=g, s=18, alpha=0.85)
        ax.set_title(f"Softmax Targeting (Neff=K={K})")
        ax.set_xlabel(r"$\hat{\psi}_{d,0}$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\mu^{adv}_d$")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(r"$g_d$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_scatter_tau_mu.png"), bbox_inches="tight")
    plt.close(fig)

def plot_mu_cdf(outdir: str, prefix: str, mu: np.ndarray, K: int):
    ensure_outdir(outdir)
    order = np.argsort(-mu)
    cumw = np.cumsum(mu[order])
    frac = (np.arange(len(mu)) + 1) / len(mu)
    fig = plt.figure(figsize=(6.2, 3.8))
    plt.plot(frac, cumw)
    plt.axhline(0.5, linestyle="--", alpha=0.6)
    plt.axhline(0.9, linestyle="--", alpha=0.6)
    plt.title(f"Attack Concentration (CDF of weights), Neff=K={K}")
    plt.xlabel("Top fraction of designs (sorted by weight)")
    plt.ylabel("Cumulative adversarial weight")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_cdf_mu.png"), bbox_inches="tight")
    plt.close(fig)

def plot_kcurve(outdir: str, prefix: str, kcurve_df: pd.DataFrame):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.3, 3.8))
    plt.plot(kcurve_df["K"], kcurve_df["S_wc"], marker="o")
    plt.title("K-curve: $K \\mapsto S^{WC}(K)$")
    plt.xlabel("K")
    plt.ylabel(r"$S^{WC}$")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_Kcurve.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# v1.1 additions: "fixed-name" plotters
# --------------------------------------------------------------------------------------

def plot_kcurve_to(path: str, kcurve_df: pd.DataFrame):
    """Save a K-curve plot to an explicit file path."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(kcurve_df["K"].values, kcurve_df["S_wc"].values, marker="o")
    ax.set_xlabel("K (effective designs)")
    ax.set_ylabel("S_wc")
    ax.set_title("K-curve")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_kcurve_split_to(path: str, kcurve_split_df: pd.DataFrame):
    """Save split/cross-fit K-curve to an explicit file path."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(
        kcurve_split_df["K"].values,
        kcurve_split_df["S_wc_split_mean"].values,
        yerr=kcurve_split_df.get("S_wc_split_se", None),
        marker="o",
        linestyle="-",
        capsize=3,
    )
    ax.set_xlabel("K (effective designs)")
    ax.set_ylabel("S_wc (split mean)")
    ax.set_title("K-curve (split/cross-fit)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_null_hist_to(path: str, null_scores: np.ndarray, s_obs: float, c_alpha: float | None = None):
    """Histogram of null scores with observed/critical lines."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(null_scores, bins=30)
    ax.axvline(s_obs, linewidth=2, linestyle="--")
    if c_alpha is not None:
        ax.axvline(c_alpha, linewidth=2, linestyle=":")
    ax.set_xlabel("S_wc under null")
    ax.set_ylabel("count")
    ax.set_title("Stage1 null distribution")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_tightening_to(path: str, tightening_df: pd.DataFrame):
    """Iteration vs bound plot for large-|D| tightening."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(tightening_df["iter"].values, tightening_df["S_wc"].values, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("S_wc")
    ax.set_title("Large-|D| tightening")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_wcurve_heat_to(path: str, w2_grid, w3_grid, wmat: np.ndarray, K: int):
    """Save w-curve heatmap to an explicit file path."""
    fig = plt.figure(figsize=(6.0, 4.1))
    plt.imshow(wmat, aspect="auto")
    plt.xticks(range(len(w3_grid)), [f"w3={x}" for x in w3_grid])
    plt.yticks(range(len(w2_grid)), [f"w2={x}" for x in w2_grid])
    plt.title(f"w-curve (S_wc), K={K}")
    plt.colorbar(label="S_wc")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_wcurve_heat(outdir: str, prefix: str, w2_grid, w3_grid, wmat: np.ndarray, K: int):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.0, 4.1))
    plt.imshow(wmat, aspect="auto")
    plt.xticks(range(len(w3_grid)), [f"w3={x}" for x in w3_grid])
    plt.yticks(range(len(w2_grid)), [f"w2={x}" for x in w2_grid])
    plt.title(f"w-curve (S_wc), K={K}")
    plt.colorbar(label=r"$S^{WC}$")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_wcurve_heat.png"), bbox_inches="tight")
    plt.close(fig)

def plot_null_hist(outdir: str, name: str, S_null: np.ndarray, S_obs: float, c_alpha: float, decision: str):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.3, 3.8))
    plt.hist(S_null, bins=18, alpha=0.85)
    plt.axvline(S_obs, linestyle="--", linewidth=2)
    plt.axvline(c_alpha, linestyle=":", linewidth=2)
    plt.title(f"Stage1 Test | {decision}")
    plt.xlabel(r"$S^{WC}$ under null")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_stage2_blocks(outdir: str, name: str, blocks_df: pd.DataFrame, K: int):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.2, 3.8))
    plt.plot(blocks_df["block"], blocks_df["S_wc"], marker="o")
    plt.title(f"Stage2 Blocked Stability (K={K})")
    plt.xlabel("block")
    plt.ylabel(r"$S^{WC}$")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_importance_ci(outdir: str, name: str, features, imp, lo, hi):
    ensure_outdir(outdir)
    order = np.argsort(-imp)
    fig = plt.figure(figsize=(6.8, 3.8))
    x = np.arange(len(features))
    plt.errorbar(x, imp[order], yerr=[imp[order]-lo[order], hi[order]-imp[order]], fmt="o")
    plt.xticks(x, np.array(features)[order], rotation=25, ha="right")
    plt.title("Stage2.5 RF Importance (bootstrap 90% band)")
    plt.ylabel("importance")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_pdp(outdir: str, name: str, grid, pdp, xlabel: str):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.2, 3.8))
    plt.plot(grid, pdp, marker="o")
    plt.title(f"PDP: {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("predicted g (avg)")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_ice(outdir: str, name: str, grid, ice_lines, pdp, xlabel: str):
    ensure_outdir(outdir)
    fig = plt.figure(figsize=(6.2, 3.8))
    for yline in ice_lines:
        plt.plot(grid, yline, alpha=0.15)
    plt.plot(grid, pdp, linewidth=2)
    plt.title(f"ICE + PDP: {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("predicted g")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# v1.1.1 additions
# -----------------------------

def save_placeholder_png(path: str, title: str, msg: str):
    """Create a small placeholder PNG so DoD file presence is guaranteed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(6.2, 3.2))
    plt.axis('off')
    plt.title(title)
    plt.text(0.01, 0.5, msg, fontsize=10, va='center')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_null_hist_multi(path: str, bundles):
    """Multi-panel null histogram for exp_root.

    bundles: list of dicts with keys {tier, S_null, S_obs, c_alpha, decision}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = max(1, len(bundles))
    ncols = 1
    nrows = n
    fig = plt.figure(figsize=(6.6, 3.2 * nrows))
    for i, b in enumerate(bundles, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        S_null = np.asarray(b.get('S_null', []), dtype=float)
        if S_null.size > 0:
            ax.hist(S_null, bins=18, alpha=0.85)
        ax.axvline(float(b.get('S_obs', np.nan)), linestyle='--', linewidth=2)
        ax.axvline(float(b.get('c_alpha', np.nan)), linestyle=':', linewidth=2)
        ax.set_title(f"{b.get('tier','')}: Stage1 Test | {b.get('decision','')}")
        ax.set_xlabel(r"$S^{WC}$ under null")
        ax.set_ylabel('count')
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
