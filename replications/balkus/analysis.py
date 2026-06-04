"""Analysis and figure generation for the Balkus replication.

Usage:
    python -m replications.balkus.analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dgps.balkus_original import generate_balkus_original

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation import compute_summary

STRAT_COLORS = {"no_cf": "#c0392b", "as_iid": "#2980b9", "chiang": "#27ae60"}
STRAT_LABELS = {"no_cf": "No CF", "as_iid": "As-IID", "chiang": "K²-fold"}
STRAT_ORDER = ["no_cf", "as_iid", "chiang"]

LEARNER_COLORS = {"earth_exact": "#8e44ad", "deep_rf": "#e74c3c", "lasso": "#f39c12"}
LEARNER_LABELS = {"earth_exact": "MARS Baseline", "deep_rf": "Deep RF", "lasso": "Lasso"}

DGP_TITLES = {
    "balkus_original":    "Balkus Original",
    "balkus_nocompress":  "No Compression",
    "shared_effects":     "Shared Effects",
    "balkus_adversarial": "Bounded Adversarial",
    "balkus_extreme":     "Bounded Extreme",
}

# Full-grid DGPs first, then adversarial (plotted separately when appropriate)
DGP_ORDER = [
    "balkus_original", "balkus_nocompress", "shared_effects",
    "balkus_adversarial", "balkus_extreme",
]


def _apply_style() -> None:
    """Set clean academic plot style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: Convergence panel ──────────────────────────────────────

def plot_convergence_panel(summary: pd.DataFrame, save_path: Path) -> None:
    """3-row × n_dgp-col panel: Bias, RMSE, Coverage vs sample size."""
    _apply_style()
    dgps = [d for d in DGP_ORDER if d in summary["dgp"].unique()]
    n_dgps = len(dgps)
    metrics = [
        ("bias", "Bias (θ̂ − θ₀)"),
        ("rmse", "RMSE"),
        ("coverage", "95% CI Coverage"),
    ]

    fig, axes = plt.subplots(3, n_dgps, figsize=(5 * n_dgps, 10),
                             sharex="col", sharey="row")
    if n_dgps == 1:
        axes = axes.reshape(-1, 1)

    for col, dgp in enumerate(dgps):
        dgp_data = summary[summary["dgp"] == dgp]

        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            learner_ls = {"earth_exact": "-", "deep_rf": "--", "lasso": ":"}
            learner_mk = {"earth_exact": "o", "deep_rf": "s", "lasso": "^"}
            for learner in ["earth_exact", "deep_rf", "lasso"]:
                ls = learner_ls[learner]
                mk = learner_mk[learner]

                for strat in STRAT_ORDER:
                    sub = dgp_data[
                        (dgp_data["learner"] == learner) &
                        (dgp_data["strategy"] == strat)
                    ].sort_values("n")
                    if len(sub) == 0:
                        continue

                    label = f"{STRAT_LABELS[strat]} ({LEARNER_LABELS[learner]})"
                    ax.plot(sub["n"], sub[metric],
                            marker=mk, color=STRAT_COLORS[strat],
                            linestyle=ls, linewidth=1.5, markersize=5,
                            label=label)

                    mcse_col = f"{metric}_mcse"
                    if mcse_col in sub.columns:
                        ax.fill_between(
                            sub["n"],
                            sub[metric] - 1.96 * sub[mcse_col],
                            sub[metric] + 1.96 * sub[mcse_col],
                            color=STRAT_COLORS[strat], alpha=0.10)

            if metric == "bias":
                ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
            elif metric == "coverage":
                ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
                ax.set_ylim(0.15, 1.05)

            if row == 0:
                ax.set_title(DGP_TITLES.get(dgp, dgp), fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("n = N × M")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Corrected AIPW: Convergence by DGP, Learner, and Strategy",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, save_path)


# ── Figure 2: Strategy comparison bars ───────────────────────────────

def plot_strategy_comparison_bars(summary: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart: Bias by strategy, faceted by DGP."""
    _apply_style()
    dgps = [d for d in DGP_ORDER if d in summary["dgp"].unique()]

    earth_data = summary[summary["learner"] == "earth_exact"].copy()
    if earth_data.empty:
        print("  Skipping strategy bars: no earth learner data")
        return

    grid_sizes = sorted(earth_data["n"].unique())
    n_grids = len(grid_sizes)

    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4), sharey=True)
    if len(dgps) == 1:
        axes = [axes]

    bar_width = 0.25
    for col, dgp in enumerate(dgps):
        ax = axes[col]
        dgp_sub = earth_data[earth_data["dgp"] == dgp]

        for i, strat in enumerate(STRAT_ORDER):
            strat_data = dgp_sub[dgp_sub["strategy"] == strat].sort_values("n")
            x_pos = np.arange(len(strat_data)) + i * bar_width

            ax.bar(x_pos, strat_data["bias"].values, bar_width,
                   color=STRAT_COLORS[strat], alpha=0.85,
                   label=STRAT_LABELS[strat])

            if "bias_mcse" in strat_data.columns:
                ax.errorbar(x_pos, strat_data["bias"].values,
                            yerr=1.96 * strat_data["bias_mcse"].values,
                            fmt="none", ecolor="black", capsize=3, linewidth=0.8)

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks(np.arange(n_grids) + bar_width)
        ax.set_xticklabels([f"{int(np.sqrt(n))}×{int(np.sqrt(n))}" for n in grid_sizes])
        ax.set_xlabel("Grid size (N × M)")
        ax.set_title(DGP_TITLES.get(dgp, dgp), fontweight="bold")
        if col == 0:
            ax.set_ylabel("Bias (θ̂ − θ₀)")

    axes[-1].legend(fontsize=9)
    fig.suptitle("True MARS: Bias by Strategy and Grid Size",
                 fontsize=12, fontweight="bold")
    _save(fig, save_path)


# ── Figure 3: MARS vs Deep RF strategy differentiation ──────────────

def plot_donsker_vs_nondonsker(summary: pd.DataFrame, save_path: Path) -> None:
    """Side-by-side: bias by strategy for each learner (balkus_original)."""
    _apply_style()

    dgp = "balkus_original"
    dgp_data = summary[summary["dgp"] == dgp].copy()
    if dgp_data.empty:
        print("  Skipping mars-vs-rf: no balkus_original data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax_idx, learner in enumerate(["earth_exact", "deep_rf"]):
        ax = axes[ax_idx]
        l_data = dgp_data[dgp_data["learner"] == learner]

        for strat in STRAT_ORDER:
            sub = l_data[l_data["strategy"] == strat].sort_values("n")
            if len(sub) == 0:
                continue

            ax.plot(sub["n"], sub["bias"],
                    marker="o", color=STRAT_COLORS[strat],
                    linewidth=2, markersize=7, label=STRAT_LABELS[strat])

            if "bias_mcse" in sub.columns:
                ax.fill_between(
                    sub["n"],
                    sub["bias"] - 1.96 * sub["bias_mcse"],
                    sub["bias"] + 1.96 * sub["bias_mcse"],
                    color=STRAT_COLORS[strat], alpha=0.12)

        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{LEARNER_LABELS[learner]}", fontweight="bold")
        ax.set_xlabel("n = N × M")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Bias (θ̂ − θ₀)")

    fig.suptitle("Balkus Original: True MARS vs Deep RF Strategy Differentiation",
                 fontsize=12, fontweight="bold")
    _save(fig, save_path)


# ── Figure 4: Donsker diagnostic ─────────────────────────────────────

def _fit_decay_slope(sizes: np.ndarray, biases: np.ndarray) -> float:
    """Fit |bias| ~ n^{-a} on log-log scale. Returns a (positive = decaying)."""
    mask = biases > 0
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(sizes[mask]), np.log(biases[mask]), 1)
    return -slope


def plot_donsker_diagnostic(summary: pd.DataFrame, save_path: Path) -> None:
    """Donsker diagnostic: per-DGP log-log bias decay + classification bars.

    Left: |bias| vs n under no_cf, one line per learner × DGP.
    Right: horizontal bar chart of fitted decay exponent a with a=0.5 threshold.
    """
    _apply_style()

    nocf = summary[summary["strategy"] == "no_cf"].copy()
    learners = [l for l in ["earth_exact", "deep_rf", "lasso"] if l in nocf["learner"].unique()]
    # Donsker diagnostic only for grid DGPs (adversarial/extreme are deep_rf-only)
    grid_dgps = ["balkus_original", "balkus_nocompress", "shared_effects"]
    dgps = [d for d in grid_dgps if d in nocf["dgp"].unique()]

    dgp_short = {"balkus_original": "Balkus Orig",
                 "balkus_nocompress": "No Compress",
                 "shared_effects": "Shared Effects"}
    dgp_linestyles = {"balkus_original": "-",
                      "balkus_nocompress": "-.", "shared_effects": ":"}
    learner_markers = {"earth_exact": "D", "deep_rf": "s", "lasso": "^"}

    fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.5),
                                           gridspec_kw={"width_ratios": [1, 1.2]})

    slope_entries = []
    for learner in learners:
        for dgp in dgps:
            sub = nocf[
                (nocf["learner"] == learner) & (nocf["dgp"] == dgp)
            ].sort_values("n")
            if sub.empty:
                continue

            sizes = sub["n"].values
            biases = np.abs(sub["bias"].values)
            slope = _fit_decay_slope(sizes, biases)

            bar_label = f"{LEARNER_LABELS[learner]} · {dgp_short[dgp]}"
            slope_entries.append((bar_label, slope, LEARNER_COLORS[learner]))

            ax_main.loglog(
                sizes, biases,
                marker=learner_markers[learner],
                color=LEARNER_COLORS[learner],
                linestyle=dgp_linestyles[dgp],
                linewidth=1.8, markersize=7, alpha=0.85,
                label=bar_label,
            )

    # n^{-1/2} reference
    n_arr = np.array([300, 3000])
    ref_half = 0.15 * (n_arr / n_arr[0]) ** (-0.5)
    ax_main.loglog(n_arr, ref_half, "--", color="gray", alpha=0.5,
                   linewidth=1.2, label=r"$n^{-1/2}$ (Donsker rate)")

    ax_main.set_xlabel("n = N × M")
    ax_main.set_ylabel("|Mean Bias|  (no cross-fitting)")
    ax_main.set_title("Bias Decay Curves (log-log)", fontweight="bold")
    ax_main.legend(fontsize=7, loc="upper right")
    ax_main.set_xlim(250, 3000)

    # Right panel: horizontal bars
    bar_labels = [e[0] for e in slope_entries]
    bar_vals = [e[1] for e in slope_entries]
    bar_colors = [e[2] for e in slope_entries]

    y_pos = np.arange(len(bar_labels))
    bars = ax_bar.barh(y_pos, bar_vals, color=bar_colors, alpha=0.75,
                       edgecolor="white", height=0.55)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=9)
    ax_bar.axvline(0.5, color="black", linewidth=1.2, linestyle="-")

    max_val = max(max(bar_vals), 0.6) if bar_vals else 1.0
    min_val = min(min(bar_vals), 0) if bar_vals else 0
    ax_bar.axvspan(min_val - 0.3, 0.5, color="#ffe0e0", alpha=0.2, zorder=0)
    ax_bar.axvspan(0.5, max_val * 1.3, color="#e0ffe0", alpha=0.2, zorder=0)

    for bar, val, color in zip(bars, bar_vals, bar_colors):
        verdict = "[PASS]" if val >= 0.5 else "[FAIL]"
        x_text = max(val + 0.03, 0.03)
        ax_bar.text(x_text, bar.get_y() + bar.get_height() / 2,
                    f"a = {val:.2f}  {verdict}",
                    va="center", fontsize=8, fontweight="bold", color=color)

    ax_bar.set_xlabel(r"Decay rate $a$  (from |bias| ~ $n^{-a}$)")
    ax_bar.set_title("Bias Decay Classification", fontweight="bold")
    ax_bar.set_xlim(min_val - 0.3, max_val * 1.3)
    ax_bar.text(0.5, -0.1, "a = 0.5 threshold", ha="center", fontsize=8,
                fontweight="bold", transform=ax_bar.get_xaxis_transform())

    fig.suptitle("Learner Stability: Bias Decay Rate Diagnostic",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, save_path)


# ── Figure 6: Disentangle narrative ──────────────────────────────────

def plot_disentangle_narrative(summary: pd.DataFrame, save_path: Path) -> None:
    """Three-panel narrative: (1) coverage without CF, (2) RMSE ratio, (3) verdict.

    Uses n=900 for the comparison.
    """
    _apply_style()

    target_n = 900
    sub = summary[summary["n"] == target_n].copy()

    dgps = [("balkus_original", "balkus"), ("shared_effects", "shared")]
    dgp_xlabels = ["Easy DGP\n(Balkus)", "Hard DGP\n(Shared)"]

    fig = plt.figure(figsize=(16, 5.5))

    # Panel 1: Coverage without cross-fitting
    ax1 = fig.add_subplot(131)
    nocf = sub[sub["strategy"] == "no_cf"]
    x = np.arange(2)
    bar_w = 0.35

    for l_idx, learner in enumerate(["earth_exact", "deep_rf"]):
        vals = []
        for dgp_full, _ in dgps:
            row = nocf[(nocf["learner"] == learner) & (nocf["dgp"] == dgp_full)]
            vals.append(row["coverage"].values[0] * 100 if len(row) > 0 else 0)

        offset = (l_idx - 0.5) * bar_w
        bars = ax1.bar(x + offset, vals, bar_w,
                       color=LEARNER_COLORS[learner], alpha=0.80,
                       label=LEARNER_LABELS[learner])
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1,
                     f"{v:.0f}%", ha="center", va="bottom", fontsize=9,
                     fontweight="bold")

    ax1.axhline(95, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax1.text(1.5, 96, "95% nominal", fontsize=8, ha="right", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dgp_xlabels)
    ax1.set_ylabel("Coverage (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("(1) Without Cross-Fitting", fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel 2: RMSE ratio (as_iid / no_cf)
    ax2 = fig.add_subplot(132)
    ratios = {}
    for learner in ["earth_exact", "deep_rf"]:
        for dgp_full, dgp_short in dgps:
            nocf_rmse = sub[(sub["learner"] == learner) & (sub["dgp"] == dgp_full) &
                            (sub["strategy"] == "no_cf")]["rmse"].values
            asiid_rmse = sub[(sub["learner"] == learner) & (sub["dgp"] == dgp_full) &
                             (sub["strategy"] == "as_iid")]["rmse"].values
            if len(nocf_rmse) > 0 and len(asiid_rmse) > 0 and nocf_rmse[0] > 0:
                ratios[(learner, dgp_short)] = asiid_rmse[0] / nocf_rmse[0]
            else:
                ratios[(learner, dgp_short)] = np.nan

    for l_idx, learner in enumerate(["earth_exact", "deep_rf"]):
        vals = [ratios.get((learner, d), np.nan) for _, d in dgps]
        offset = (l_idx - 0.5) * bar_w
        bars = ax2.bar(x + offset, vals, bar_w,
                       color=LEARNER_COLORS[learner], alpha=0.80,
                       label=LEARNER_LABELS[learner])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         min(v + 0.3, 22), f"{v:.1f}×",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.text(1.5, 1.1, "No change", fontsize=8, ha="right", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dgp_xlabels)
    ax2.set_ylabel("RMSE Ratio (As-IID / No-CF)")
    ax2.set_yscale("symlog", linthresh=2)
    ax2.set_ylim(0, 25)
    ax2.set_title("(2) Does Cross-Fitting Help?", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")

    # Panel 3: Verdict text
    ax3 = fig.add_subplot(133)
    ax3.axis("off")

    mars_cov = {}
    rf_cov = {}
    for dgp_full, dgp_short in dgps:
        row_m = nocf[(nocf["learner"] == "earth_exact") & (nocf["dgp"] == dgp_full)]
        row_r = nocf[(nocf["learner"] == "deep_rf") & (nocf["dgp"] == dgp_full)]
        mars_cov[dgp_short] = int(row_m["coverage"].values[0] * 100) if len(row_m) > 0 else 0
        rf_cov[dgp_short] = int(row_r["coverage"].values[0] * 100) if len(row_r) > 0 else 0

    mars_ratio_easy = ratios.get(("earth_exact", "balkus"), 0)
    mars_ratio_hard = ratios.get(("earth_exact", "shared"), 0)

    verdict_text = (
        "Verdict\n"
        "========================\n\n"
        "MARS without CF:\n"
        f"  Easy DGP: {mars_cov.get('balkus', '?')}% coverage  "
        f"{'OK' if mars_cov.get('balkus', 0) >= 80 else 'FAIL'}\n"
        f"  Hard DGP: {mars_cov.get('shared', '?')}% coverage  "
        f"{'OK' if mars_cov.get('shared', 0) >= 80 else 'FAIL'}\n"
        "  -> Works on both DGPs\n"
        f"  -> CF hurts ({mars_ratio_easy:.0f}-{mars_ratio_hard:.0f}x RMSE)\n\n"
        "Deep RF without CF:\n"
        f"  Easy DGP: {rf_cov.get('balkus', '?')}% coverage  "
        f"{'OK' if rf_cov.get('balkus', 0) >= 80 else 'FAIL'}\n"
        f"  Hard DGP: {rf_cov.get('shared', '?')}%  coverage  "
        f"{'OK' if rf_cov.get('shared', 0) >= 80 else 'FAIL'}\n"
        "  -> Fails on both DGPs\n"
        "  -> Cross-fitting fixes it\n\n"
        "========================\n\n"
        "Conclusion:\n"
        "The pattern is a learner\n"
        "property, not a DGP artifact.\n\n"
        "MARS is stable without CF.\n"
        "Deep RF requires CF to\n"
        "achieve valid coverage."
    )

    ax3.text(0.05, 0.95, verdict_text, transform=ax3.transAxes,
             fontsize=10, fontfamily="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa",
                       edgecolor="#555555", linewidth=1))
    ax3.set_title("(3) Interpretation", fontweight="bold")

    fig.suptitle("Disentangling Experiment: Is It the Learner or the DGP?",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── Figure 7: DGP heatmaps ───────────────────────────────────────────

def plot_dgp_heatmaps(save_path: Path, N: int = 40, M: int = 40) -> None:
    """2×3 heatmap panel visualizing the internal structure of the Balkus DGP.

    Generates a single draw from the DGP and reshapes the flat observation
    vector into an (N, M) grid. Each panel reveals a different structural
    component: baseline outcome, covariate, propensity, treatment effect,
    treatment assignment, and observed outcome.

    Args:
        save_path: Where to save the figure.
        N: Number of row clusters for the visualization draw.
        M: Number of column clusters for the visualization draw.
    """
    _apply_style()

    df, ate = generate_balkus_original(N=N, M=M, seed=42)

    panels = [
        ("mu0_true", "viridis",  "Baseline Outcome ($Q_0$)",
         "Additive row/col cluster banding"),
        ("X1",       "coolwarm", "Covariate $X_1$",
         "Row + column effects ($a_X + b_X$)"),
        ("pA_true",  "magma",    "Propensity Score ($p_A$)",
         "Logit-compressed cluster pattern"),
        ("tau_true", "plasma",   "Treatment Effect ($\\tau_{ij}$)",
         "Heterogeneous, non-linear interactions"),
        ("A",        "Greys",    "Treatment Assignment ($A$)",
         "Bernoulli draws biased by $p_A$"),
        ("Y",        "inferno",  "Observed Outcome ($Y$)",
         "$Q_0 + A \\cdot \\tau + \\varepsilon$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Balkus et al. (2026) Two-Way Clustered DGP  "
        f"({N}×{M} grid, ATE = {ate:.2f})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for ax, (col, cmap, title, subtitle) in zip(axes.flat, panels):
        mat = df[col].values.reshape(N, M)
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("Column Clusters (M)")
        ax.set_ylabel("Row Clusters (N)")
        fig.colorbar(im, ax=ax)

    _save(fig, save_path)


# ── Figure 7: IID vs Cluster-Robust SE Coverage ─────────────────────

def plot_iid_vs_cr_coverage(df: pd.DataFrame, save_path: Path) -> None:
    """Coverage comparison: IID SE vs Cluster-Robust SE, by DGP and strategy.

    Uses raw data with covers_iid and covers_cr columns.
    """
    _apply_style()

    has_cr = df["covers_cr"].notna()
    has_iid = df["covers_iid"].notna()
    sub = df[has_cr & has_iid].copy()
    if sub.empty:
        print("  Skipping IID vs CR: no cluster-robust data")
        return

    if "n" not in sub.columns:
        sub["n"] = sub["N"] * sub["M"]

    dgps = [d for d in DGP_ORDER if d in sub["dgp"].unique()]
    n_dgps = len(dgps)

    fig, axes = plt.subplots(1, n_dgps, figsize=(5 * n_dgps, 5), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    for col_idx, dgp in enumerate(dgps):
        ax = axes[col_idx]
        dgp_sub = sub[sub["dgp"] == dgp]

        g = dgp_sub.groupby(["learner", "strategy", "n"]).agg(
            cov_iid=("covers_iid", "mean"),
            cov_cr=("covers_cr", "mean"),
        ).reset_index()

        for learner in ["deep_rf", "earth_exact", "lasso"]:
            if learner not in g["learner"].unique():
                continue
            for strat in ["as_iid", "chiang"]:
                ls = g[(g["learner"] == learner) & (g["strategy"] == strat)].sort_values("n")
                if ls.empty:
                    continue

                label = f"{STRAT_LABELS[strat]} ({LEARNER_LABELS[learner]})"
                ax.scatter(ls["cov_iid"], ls["cov_cr"],
                           color=STRAT_COLORS[strat], marker="o",
                           s=60, alpha=0.8, label=label, zorder=3)

        ax.plot([0.5, 1.05], [0.5, 1.05], "k--", linewidth=0.8, alpha=0.5)
        ax.axhline(0.95, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axvline(0.95, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xlabel("Coverage (IID SE)")
        ax.set_title(DGP_TITLES.get(dgp, dgp), fontweight="bold")
        ax.set_xlim(0.55, 1.05)
        ax.set_ylim(0.55, 1.05)

        if col_idx == 0:
            ax.set_ylabel("Coverage (Cluster-Robust SE)")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("IID vs Cluster-Robust SE: Coverage Comparison",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, save_path)


# ── Figure 8: Scorecard heatmap ──────────────────────────────────────

def plot_scorecard_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Heatmap: coverage difference (as_iid - chiang) across all cells."""
    _apply_style()

    if "n" not in df.columns:
        df = df.copy()
        df["n"] = df["N"] * df["M"]

    g = df.groupby(["dgp", "learner", "N", "strategy"]).agg(
        cov=("covers", "mean"),
    ).reset_index()

    pivot = g.pivot_table(
        index=["dgp", "learner", "N"],
        columns="strategy",
        values="cov",
    ).reset_index()

    if "as_iid" not in pivot.columns or "chiang" not in pivot.columns:
        print("  Skipping scorecard: missing strategies")
        return

    pivot["delta"] = pivot["as_iid"] - pivot["chiang"]

    dgps = [d for d in DGP_ORDER if d in pivot["dgp"].unique()]
    learners = ["earth_exact", "deep_rf", "lasso"]
    Ns = sorted(pivot["N"].unique())

    n_rows = len(learners) * len(Ns)
    n_cols = len(dgps)

    fig, ax = plt.subplots(figsize=(3 * n_cols + 2, 0.5 * n_rows + 2))

    matrix = np.full((n_rows, n_cols), np.nan)
    y_labels = []

    for r_idx, (learner, N) in enumerate(
        [(l, n) for l in learners for n in Ns]
    ):
        y_labels.append(f"{LEARNER_LABELS.get(learner, learner)} N={N}")
        for c_idx, dgp in enumerate(dgps):
            row = pivot[
                (pivot["dgp"] == dgp) &
                (pivot["learner"] == learner) &
                (pivot["N"] == N)
            ]
            if len(row) > 0:
                matrix[r_idx, c_idx] = row["delta"].values[0]

    im = ax.imshow(matrix, cmap="RdBu", aspect="auto",
                   vmin=-0.15, vmax=0.15)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                txt = f"{val:+.0%}"
                color = "white" if abs(val) > 0.10 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([DGP_TITLES.get(d, d) for d in dgps], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)

    fig.colorbar(im, ax=ax, label="Coverage: As-IID minus K²-fold",
                 shrink=0.7)

    # Add horizontal separators between learners
    for i in range(1, len(learners)):
        ax.axhline(i * len(Ns) - 0.5, color="black", linewidth=1.5)

    ax.set_title("Scorecard: As-IID vs K²-fold Coverage Difference\n"
                 "(Blue = As-IID better, Red = K²-fold better)",
                 fontweight="bold", fontsize=11)
    _save(fig, save_path)


# ── Figure 9: Morris et al. (2019) RelSE Diagnostic ─────────────────

def plot_morris_relse(df: pd.DataFrame, save_path: Path) -> None:
    """RelSE comparison: IID SE vs Cluster-Robust SE.

    Left: scatter of RelSE(IID) vs RelSE(CR) — points above diagonal
    mean CR is more conservative; points near RelSE=1 are well-calibrated.
    Right: grouped bars showing RelSE by learner × strategy for each SE type.
    """
    _apply_style()

    has_cr = df["se_cr"].notna() & df["covers_cr"].notna()
    sub = df[has_cr].copy()
    if sub.empty:
        print("  Skipping Morris RelSE: no cluster-robust SE data")
        return

    if "n" not in sub.columns:
        sub["n"] = sub["N"] * sub["M"]

    group_cols = ["dgp", "learner", "strategy", "n"]

    # Compute Morris metrics for IID and CR SEs
    summary_iid = compute_summary(sub, group_cols=group_cols,
                                  se_col="se_iid", covers_col="covers_iid")
    summary_cr = compute_summary(sub, group_cols=group_cols,
                                 se_col="se_cr", covers_col="covers_cr")

    merged = summary_iid[group_cols + ["rel_se", "coverage"]].rename(
        columns={"rel_se": "rel_se_iid", "coverage": "cov_iid"}
    ).merge(
        summary_cr[group_cols + ["rel_se", "coverage"]].rename(
            columns={"rel_se": "rel_se_cr", "coverage": "cov_cr"}
        ),
        on=group_cols,
    )

    # Save the full Morris comparison table
    merged.to_csv(save_path.parent / "morris_relse_comparison.csv", index=False)
    print(f"  Saved: {save_path.parent / 'morris_relse_comparison.csv'}")

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter of RelSE(IID) vs RelSE(CR)
    learner_markers = {"earth_exact": "D", "deep_rf": "s", "lasso": "^"}

    for _, row in merged.iterrows():
        strat = row["strategy"]
        learner = row["learner"]
        ax_scatter.scatter(
            row["rel_se_iid"], row["rel_se_cr"],
            color=STRAT_COLORS.get(strat, "gray"),
            marker=learner_markers.get(learner, "o"),
            s=80, alpha=0.7, zorder=3,
        )

    # Reference lines
    ax_scatter.plot([0, 2], [0, 2], "k--", linewidth=0.8, alpha=0.4)
    ax_scatter.axhline(1.0, color="green", linewidth=1.2, linestyle=":",
                       alpha=0.5)
    ax_scatter.axvline(1.0, color="green", linewidth=1.2, linestyle=":",
                       alpha=0.5)

    # Shade calibration zone
    ax_scatter.axhspan(0.9, 1.1, color="green", alpha=0.04)
    ax_scatter.axvspan(0.9, 1.1, color="green", alpha=0.04)

    ax_scatter.set_xlabel("RelSE (IID SE)")
    ax_scatter.set_ylabel("RelSE (Cluster-Robust SE)")
    ax_scatter.set_title("A. SE Calibration: IID vs CR", fontweight="bold")
    ax_scatter.set_xlim(0.1, 1.5)
    ax_scatter.set_ylim(0.1, 1.5)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="D", color="gray", markersize=7,
               linestyle="None", label="MARS"),
        Line2D([0], [0], marker="s", color="gray", markersize=7,
               linestyle="None", label="Deep RF"),
        Line2D([0], [0], marker="^", color="gray", markersize=7,
               linestyle="None", label="Lasso"),
        Line2D([0], [0], marker="o", color=STRAT_COLORS["no_cf"],
               markersize=7, linestyle="None", label="No CF"),
        Line2D([0], [0], marker="o", color=STRAT_COLORS["as_iid"],
               markersize=7, linestyle="None", label="As-IID"),
        Line2D([0], [0], marker="o", color=STRAT_COLORS["chiang"],
               markersize=7, linestyle="None", label="K²-fold"),
    ]
    ax_scatter.legend(handles=legend_handles, fontsize=8, loc="upper left")

    # Panel B: Bar chart — mean RelSE by (learner, strategy)
    agg = merged.groupby(["learner", "strategy"]).agg(
        relse_iid=("rel_se_iid", "mean"),
        relse_cr=("rel_se_cr", "mean"),
    ).reset_index()

    learners = ["earth_exact", "deep_rf", "lasso"]
    strats = STRAT_ORDER
    x_labels = []
    iid_vals = []
    cr_vals = []
    bar_colors = []

    for learner in learners:
        for strat in strats:
            row = agg[(agg["learner"] == learner) & (agg["strategy"] == strat)]
            if row.empty:
                continue
            x_labels.append(
                f"{LEARNER_LABELS.get(learner, learner)}\n{STRAT_LABELS[strat]}")
            iid_vals.append(row["relse_iid"].values[0])
            cr_vals.append(row["relse_cr"].values[0])
            bar_colors.append(STRAT_COLORS[strat])

    x = np.arange(len(x_labels))
    w = 0.35
    bars_iid = ax_bar.bar(x - w / 2, iid_vals, w, color="lightcoral",
                          alpha=0.7, label="IID SE", edgecolor="white")
    bars_cr = ax_bar.bar(x + w / 2, cr_vals, w, color="steelblue",
                         alpha=0.7, label="CR SE", edgecolor="white")

    ax_bar.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
                   alpha=0.7)
    ax_bar.axhspan(0.9, 1.1, color="green", alpha=0.05)

    for bar, val in zip(bars_iid, iid_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                    color="firebrick")
    for bar, val in zip(bars_cr, cr_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                    color="steelblue")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax_bar.set_ylabel("RelSE (ModSE / EmpSE)")
    ax_bar.set_title("B. Mean RelSE by Learner × Strategy", fontweight="bold")
    ax_bar.set_ylim(0, 1.3)
    ax_bar.legend(fontsize=9, loc="upper right")

    # Separator lines between learners
    for i in range(1, len(learners)):
        sep_x = i * len(strats) - 0.5
        if sep_x < len(x_labels):
            ax_bar.axvline(sep_x, color="gray", linewidth=0.5,
                           linestyle=":", alpha=0.5)

    fig.suptitle(
        "Morris et al. (2019) SE Calibration: IID vs Cluster-Robust",
        fontsize=13, fontweight="bold", y=1.01)
    _save(fig, save_path)


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    """Generate all figures for the Balkus replication."""
    results_csv = Path(__file__).parent / "results" / "balkus_results.csv"

    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found. Run main.py first.")
        return

    plots_dir = results_csv.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    # Drop balkus_hard — excluded from the report
    df = df[df["dgp"] != "balkus_hard"].copy()
    if "n" not in df.columns:
        df["n"] = df["N"] * df["M"]

    print(f"Loaded {len(df)} rows from {results_csv} (balkus_hard excluded)")
    summary = compute_summary(df, group_cols=["dgp", "learner", "strategy", "n"])
    print(f"Computed summary: {len(summary)} scenario groups\n")

    summary.to_csv(results_csv.parent / "balkus_summary.csv", index=False)
    print(f"  Summary saved to balkus_summary.csv")

    plot_convergence_panel(summary, plots_dir / "convergence_panel.png")
    plot_strategy_comparison_bars(summary, plots_dir / "strategy_bars.png")
    plot_donsker_vs_nondonsker(summary, plots_dir / "mars_vs_deep_rf.png")

    plot_donsker_diagnostic(summary, plots_dir / "donsker_diagnostic.png")
    plot_disentangle_narrative(summary, plots_dir / "disentangle_narrative.png")
    plot_dgp_heatmaps(plots_dir / "dgp_heatmaps.png")
    plot_iid_vs_cr_coverage(df, plots_dir / "iid_vs_cr_coverage.png")
    plot_scorecard_heatmap(df, plots_dir / "scorecard_heatmap.png")
    plot_morris_relse(df, plots_dir / "morris_relse.png")

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()


