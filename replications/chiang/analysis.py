"""Analysis and figure generation for Chiang et al. (2021) and Chen & Chiang (2026)."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation import compute_summary

STRAT_COLORS: Dict[str, str] = {
    "no_cf": "#c0392b",
    "as_iid": "#2980b9",
    "chiang_k2": "#27ae60",
}
STRAT_LABELS: Dict[str, str] = {
    "no_cf": "No CF",
    "as_iid": "As-IID",
    "chiang_k2": "K²-fold",
}
STRAT_ORDER: List[str] = ["no_cf", "as_iid", "chiang_k2"]


def _apply_style() -> None:
    """Set consistent publication-quality style for plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_chiang_calibration(results_csv: Path, save_path: Path) -> None:
    """Plot Coverage and RelSE across Chiang et al. (2021) scenarios.

    Args:
        results_csv: Path to the raw simulation results CSV.
        save_path: Path where the generated figure should be saved.

    Returns:
        None.
    """
    _apply_style()
    df = pd.read_csv(results_csv)
    
    # Performance measures: Bias, EmpSE, ModSE, RelSE, Coverage — Morris et al. (2019)
    summary = compute_summary(df, group_cols=["scenario", "strategy"])

    scenario_clean_labels = {
        "(25,25), dim=100, K^2=4, Lasso": "(25,25)\nP=100, K²=4",
        "(25,25), dim=200, K^2=4, Lasso": "(25,25)\nP=200, K²=4",
        "(25,25), dim=100, K^2=9, Lasso": "(25,25)\nP=100, K²=9",
        "(25,25), dim=200, K^2=9, Lasso": "(25,25)\nP=200, K²=9",
        "(50,50), dim=100, K^2=4, Lasso": "(50,50)\nP=100, K²=4",
        "(50,50), dim=200, K^2=4, Lasso": "(50,50)\nP=200, K²=4",
    }
    
    scenarios_ordered = list(scenario_clean_labels.keys())
    
    fig, (ax_cov, ax_relse) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    n_scens = len(scenarios_ordered)
    n_strats = len(STRAT_ORDER)
    width = 0.8 / n_strats
    offsets = np.arange(n_strats) - (n_strats - 1) / 2
    
    for s_idx, strat in enumerate(STRAT_ORDER):
        cov_rates = []
        relse_vals = []
        cov_mcses = []
        relse_mcses = []
        
        for scen in scenarios_ordered:
            sub = summary[(summary["scenario"] == scen) & (summary["strategy"] == strat)]
            if len(sub) > 0:
                row = sub.iloc[0]
                cov_rates.append(row["coverage"])
                cov_mcses.append(row.get("coverage_mcse", 0))
                # Fallback to 0 if standard error is invalid due to variance estimation collapse
                relse_val = row["rel_se"] if not np.isnan(row["rel_se"]) else 0.0
                relse_vals.append(relse_val)
                relse_mcses.append(row.get("rel_se_mcse", 0))
            else:
                cov_rates.append(0.0)
                cov_mcses.append(0.0)
                relse_vals.append(0.0)
                relse_mcses.append(0.0)
                
        x = np.arange(n_scens) + offsets[s_idx] * width
        
        ax_cov.bar(x, cov_rates, width, color=STRAT_COLORS[strat],
                   label=STRAT_LABELS[strat], edgecolor="white", linewidth=0.5)
        ax_cov.errorbar(x, cov_rates, yerr=1.96 * np.array(cov_mcses),
                        fmt="none", color="black", capsize=2, linewidth=0.6)
        
        ax_relse.bar(x, relse_vals, width, color=STRAT_COLORS[strat],
                     label=STRAT_LABELS[strat], edgecolor="white", linewidth=0.5)
        ax_relse.errorbar(x, relse_vals, yerr=1.96 * np.array(relse_mcses),
                          fmt="none", color="black", capsize=2, linewidth=0.6)

    # Reference lines and intervals: Morris et al. (2019) performance standards
    ax_cov.axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_cov.axhspan(0.90, 1.0, color="green", alpha=0.05)
    ax_cov.set_ylabel("95% CI Coverage")
    ax_cov.set_title("95% Nominal Coverage Rates (±95% MCSE)")
    ax_cov.set_ylim(0, 1.1)
    
    ax_relse.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_relse.axhspan(0.9, 1.1, color="green", alpha=0.05)
    ax_relse.set_ylabel("RelSE (ModSE / EmpSE)")
    ax_relse.set_title("SE Calibration: RelSE (values < 1 → anti-conservative)")
    ax_relse.set_ylim(0, 1.5)
    
    ax_relse.set_xticks(range(n_scens))
    ax_relse.set_xticklabels([scenario_clean_labels[s] for s in scenarios_ordered])
    ax_relse.set_xlabel("Simulation Scenario (Grid Size, Covariates, Base Folds)")
    
    ax_cov.legend(loc="lower right")
    
    fig.suptitle("Chiang et al. (2021) Replication Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_chen_chiang_convergence(results_csv: Path, save_path: Path) -> None:
    """Plot coverage convergence vs grid size for Chen & Chiang (2026).

    Args:
        results_csv: Path to the raw simulation results CSV.
        save_path: Path where the generated figure should be saved.

    Returns:
        None.
    """
    _apply_style()
    df = pd.read_csv(results_csv)
    
    # Calculate sample size for two-way grid setup (n = N * M)
    if "n" not in df.columns:
        df["n"] = df["N"] * df["M"]
        
    summary = compute_summary(df, group_cols=["strategy", "n"])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for strat in STRAT_ORDER:
        sub = summary[summary["strategy"] == strat].sort_values("n")
        if len(sub) == 0:
            continue
            
        ax.plot(sub["n"], sub["coverage"], marker="o", color=STRAT_COLORS[strat],
                linewidth=2.0, label=STRAT_LABELS[strat])
        ax.fill_between(sub["n"], sub["coverage"] - 1.96 * sub["coverage_mcse"],
                        sub["coverage"] + 1.96 * sub["coverage_mcse"],
                        color=STRAT_COLORS[strat], alpha=0.1)

    # Reference boundaries for nominal coverage rates
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Nominal 95%")
    ax.axhspan(0.90, 1.0, color="green", alpha=0.05)
    
    ax.set_xlabel("Sample size (n = N × M)")
    ax.set_ylabel("95% CI Coverage")
    ax.set_title("Chen & Chiang (2026) Convergence Validation\n(No-CF DML2 Convergence)", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)
    
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_doubleml_comparison(results_csv: Path, save_path: Path) -> None:
    """2x2 diagnostic plot: Chiang K²-fold vs as-IID on DoubleML DGP.

    Panels: (A) violin of point estimates, (B) SE calibration scatter with
    RelSE annotation, (C) Chiang caterpillar CIs, (D) As-IID caterpillar CIs.

    Args:
        results_csv: Path to per-replication CSV with theta_*/se_* columns.
        save_path: Output path for the figure.
    """
    _apply_style()
    import matplotlib.gridspec as gridspec

    THETA_TRUE = 1.0
    df = pd.read_csv(results_csv)

    emp_se_chiang = df["theta_chiang"].std()
    emp_se_iid = df["theta_iid"].std()
    mean_mod_se_chiang = df["se_chiang"].mean()
    mean_mod_se_iid = df["se_iid"].mean()

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    colors = {"chiang": "#c0392b", "iid": "#2980b9"}

    # Panel A: Distribution of theta_hat
    ax1 = fig.add_subplot(gs[0, 0])
    parts = ax1.violinplot(
        [df["theta_chiang"], df["theta_iid"]],
        positions=[1, 2], showmedians=True, showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(colors.values())[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax1.axhline(THETA_TRUE, color="grey", ls="--", lw=1, label=r"$\theta_0 = 1$")
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Chiang $K^2$", "As-IID"])
    ax1.set_ylabel(r"$\hat{\theta}$")
    ax1.set_title("A. Distribution of Point Estimates")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.text(1, ax1.get_ylim()[0] + 0.02, f"EmpSE={emp_se_chiang:.3f}",
             ha="center", fontsize=9, color=colors["chiang"])
    ax1.text(2, ax1.get_ylim()[0] + 0.02, f"EmpSE={emp_se_iid:.3f}",
             ha="center", fontsize=9, color=colors["iid"])

    # Panel B: Model SE vs Empirical SE — Morris et al. (2019, Section 5.4)
    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(42)
    ax2.scatter(np.ones(len(df)) + rng.normal(0, 0.05, len(df)),
                df["se_chiang"], alpha=0.3, s=15, c=colors["chiang"])
    ax2.scatter(2 * np.ones(len(df)) + rng.normal(0, 0.05, len(df)),
                df["se_iid"], alpha=0.3, s=15, c=colors["iid"])
    ax2.hlines(emp_se_chiang, 0.6, 1.4, color=colors["chiang"], ls="--", lw=2)
    ax2.hlines(emp_se_iid, 1.6, 2.4, color=colors["iid"], ls="--", lw=2)

    rel_chiang = mean_mod_se_chiang / emp_se_chiang
    rel_iid = mean_mod_se_iid / emp_se_iid
    pct_chiang = (rel_chiang - 1) * 100
    pct_iid = (1 - rel_iid) * 100
    ax2.annotate(f"RelSE={rel_chiang:.2f}\n({pct_chiang:.0f}% too large)",
                 xy=(1, mean_mod_se_chiang), xytext=(0.5, 0.15),
                 textcoords="axes fraction", fontsize=9, color=colors["chiang"],
                 fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=colors["chiang"], lw=1.2))
    ax2.annotate(f"RelSE={rel_iid:.2f}\n({pct_iid:.0f}% too small)",
                 xy=(2, mean_mod_se_iid), xytext=(0.75, 0.85),
                 textcoords="axes fraction", fontsize=9, color=colors["iid"],
                 fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=colors["iid"], lw=1.2))
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Chiang $K^2$", "As-IID"])
    ax2.set_ylabel("Model SE (dots) vs EmpSE (dashed)")
    ax2.set_title("B. SE Calibration: RelSE = ModSE / EmpSE")
    ax2.set_xlim(0.4, 2.6)

    # Panels C & D: Caterpillar CI plots
    covers_chiang = (
        (df["theta_chiang"] - 1.96 * df["se_chiang"] <= THETA_TRUE)
        & (THETA_TRUE <= df["theta_chiang"] + 1.96 * df["se_chiang"])
    )
    covers_iid = (
        (df["theta_iid"] - 1.96 * df["se_iid"] <= THETA_TRUE)
        & (THETA_TRUE <= df["theta_iid"] + 1.96 * df["se_iid"])
    )

    idx_c = df["theta_chiang"].argsort().values
    idx_i = df["theta_iid"].argsort().values

    ax3 = fig.add_subplot(gs[1, 0])
    for rank, i in enumerate(idx_c[:50]):
        lo = df.iloc[i]["theta_chiang"] - 1.96 * df.iloc[i]["se_chiang"]
        hi = df.iloc[i]["theta_chiang"] + 1.96 * df.iloc[i]["se_chiang"]
        color = colors["chiang"] if covers_chiang.iloc[i] else "grey"
        ax3.plot([lo, hi], [rank, rank], color=color, alpha=0.5, lw=1.2)
    ax3.axvline(THETA_TRUE, color="black", ls="--", lw=1)
    ax3.set_xlabel(r"$\hat{\theta} \pm 1.96 \cdot SE$")
    ax3.set_ylabel("Replication (sorted)")
    ax3.set_title(f"C. Chiang $K^2$ CIs (Coverage = {covers_chiang.mean()*100:.0f}%)")
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    for rank, i in enumerate(idx_i[:50]):
        lo = df.iloc[i]["theta_iid"] - 1.96 * df.iloc[i]["se_iid"]
        hi = df.iloc[i]["theta_iid"] + 1.96 * df.iloc[i]["se_iid"]
        color = colors["iid"] if covers_iid.iloc[i] else "grey"
        ax4.plot([lo, hi], [rank, rank], color=color, alpha=0.5, lw=1.2)
    ax4.axvline(THETA_TRUE, color="black", ls="--", lw=1)
    ax4.set_xlabel(r"$\hat{\theta} \pm 1.96 \cdot SE$")
    ax4.set_ylabel("Replication (sorted)")
    ax4.set_title(f"D. As-IID CIs (Coverage = {covers_iid.mean()*100:.0f}%)")
    ax4.set_yticks([])

    xlim = (
        min(ax3.get_xlim()[0], ax4.get_xlim()[0]),
        max(ax3.get_xlim()[1], ax4.get_xlim()[1]),
    )
    ax3.set_xlim(xlim)
    ax4.set_xlim(xlim)

    fig.suptitle(
        "DoubleML Head-to-Head: Chiang $K^2$-fold vs As-IID\n"
        "Official Chiang DGP ($N$=$M$=25, $p$=100, $K$=3)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> None:
    """Process all Chiang replication results and generate plots."""
    parser = argparse.ArgumentParser(description="Analyze Chiang and Chen-Chiang simulation results.")
    parser.add_argument("--chiang-csv", type=str,
                        default=str(_PROJECT_ROOT / "replications" / "chiang" / "results" / "chiang_replication.csv"))
    parser.add_argument("--chen-csv", type=str,
                        default=str(_PROJECT_ROOT / "replications" / "chiang" / "results" / "chen_chiang_validation.csv"))
    parser.add_argument("--doubleml-csv", type=str,
                        default=str(_PROJECT_ROOT / "replications" / "chiang" / "results" / "doubleml_head_to_head_results.csv"))
    args = parser.parse_args()
    
    results_dir = _PROJECT_ROOT / "replications" / "chiang" / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    chiang_path = Path(args.chiang_csv)
    chen_path = Path(args.chen_csv)
    doubleml_path = Path(args.doubleml_csv)
    
    if chiang_path.exists():
        print(f"Processing Chiang results from: {chiang_path}")
        plot_chiang_calibration(chiang_path, plots_dir / "chiang_calibration.png")
        
        df_chiang = pd.read_csv(chiang_path)
        summary_chiang = compute_summary(df_chiang, group_cols=["scenario", "strategy"])
        summary_chiang.to_csv(results_dir / "chiang_summary_metrics.csv", index=False)
        print(f"  Saved summary: {results_dir / 'chiang_summary_metrics.csv'}")
    else:
        print(f"WARNING: Chiang results not found at {chiang_path}")
        
    if chen_path.exists():
        print(f"Processing Chen & Chiang results from: {chen_path}")
        plot_chen_chiang_convergence(chen_path, plots_dir / "chen_chiang_convergence.png")
        
        df_chen = pd.read_csv(chen_path)
        summary_chen = compute_summary(df_chen, group_cols=["strategy", "n"])
        summary_chen.to_csv(results_dir / "chen_chiang_summary_metrics.csv", index=False)
        print(f"  Saved summary: {results_dir / 'chen_chiang_summary_metrics.csv'}")
    else:
        print(f"Note: Chen & Chiang convergence results not found at {chen_path} (yet to be run).")

    # Also check for DoubleML head-to-head results in scratch/ (legacy) or results/
    if not doubleml_path.exists():
        doubleml_path = _PROJECT_ROOT / "scratch" / "doubleml_head_to_head_results.csv"
    if doubleml_path.exists():
        print(f"Processing DoubleML head-to-head results from: {doubleml_path}")
        plot_doubleml_comparison(doubleml_path, results_dir / "doubleml_comparison.png")
    else:
        print(f"Note: DoubleML head-to-head results not found (run doubleml_validation.py first).")


if __name__ == "__main__":
    main()

