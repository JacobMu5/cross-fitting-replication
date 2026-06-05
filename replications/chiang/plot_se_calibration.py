"""SE calibration scatter: IID vs CR SEs across strategies on Chiang's DGP."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def plot_se_calibration_dual(results_csv: Path, save_path: Path,
                             scenario: str = "(25,25), dim=100, K^2=4, Lasso") -> None:
    """4-column scatter: (Chiang IID, Chiang CR) vs (As-IID IID, As-IID CR).

    Shows how CR SEs close the gap between ModSE and EmpSE.

    Args:
        results_csv: Path to chiang_replication.csv with dual-SE columns.
        save_path: Output path for the figure.
        scenario: Which scenario to plot.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    df = pd.read_csv(results_csv)
    df = df[df["scenario"] == scenario]

    chiang = df[df["strategy"] == "chiang_k2"]
    iid = df[df["strategy"] == "as_iid"]

    emp_se_chiang = chiang["tau_hat"].std()
    emp_se_iid = iid["tau_hat"].std()

    colors = {
        "chiang_iid": "#e74c3c",
        "chiang_cr": "#c0392b",
        "iid_iid": "#3498db",
        "iid_cr": "#2471a3",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    positions = {
        "chiang_iid": 1,
        "chiang_cr": 2,
        "iid_iid": 3.5,
        "iid_cr": 4.5,
    }

    # Chiang K²: IID SEs
    ax.scatter(
        positions["chiang_iid"] + rng.normal(0, 0.06, len(chiang)),
        chiang["se_iid"], alpha=0.35, s=18, c=colors["chiang_iid"],
        label=r"Chiang $K^2$ — IID SE", zorder=3,
    )
    # Chiang K²: CR SEs
    ax.scatter(
        positions["chiang_cr"] + rng.normal(0, 0.06, len(chiang)),
        chiang["se_cr"], alpha=0.35, s=18, c=colors["chiang_cr"],
        label=r"Chiang $K^2$ — CR SE", zorder=3, marker="D",
    )
    # As-IID: IID SEs
    ax.scatter(
        positions["iid_iid"] + rng.normal(0, 0.06, len(iid)),
        iid["se_iid"], alpha=0.35, s=18, c=colors["iid_iid"],
        label="As-IID — IID SE", zorder=3,
    )
    # As-IID: CR SEs
    ax.scatter(
        positions["iid_cr"] + rng.normal(0, 0.06, len(iid)),
        iid["se_cr"], alpha=0.35, s=18, c=colors["iid_cr"],
        label="As-IID — CR SE", zorder=3, marker="D",
    )

    # EmpSE reference lines
    ax.hlines(emp_se_chiang, 0.5, 2.5, color=colors["chiang_iid"],
              ls="--", lw=2.0, alpha=0.8)
    ax.hlines(emp_se_iid, 3.0, 5.0, color=colors["iid_iid"],
              ls="--", lw=2.0, alpha=0.8)

    # RelSE annotations
    mean_chiang_iid = chiang["se_iid"].mean()
    mean_chiang_cr = chiang["se_cr"].mean()
    mean_iid_iid = iid["se_iid"].mean()
    mean_iid_cr = iid["se_cr"].mean()

    rel_chiang_iid = mean_chiang_iid / emp_se_chiang
    rel_chiang_cr = mean_chiang_cr / emp_se_chiang
    rel_iid_iid = mean_iid_iid / emp_se_iid
    rel_iid_cr = mean_iid_cr / emp_se_iid

    def _annotate(x: float, y: float, rel: float, color: str, ha: str = "center") -> None:
        pct = abs(1 - rel) * 100
        direction = "too small" if rel < 1 else "too large"
        ax.annotate(
            f"RelSE={rel:.2f}\n({pct:.0f}% {direction})",
            xy=(x, y), xytext=(x, y - 0.012),
            fontsize=9, color=color, fontweight="bold", ha=ha,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    _annotate(1.0, mean_chiang_iid, rel_chiang_iid, colors["chiang_iid"])
    _annotate(2.0, mean_chiang_cr, rel_chiang_cr, colors["chiang_cr"])
    _annotate(3.5, mean_iid_iid, rel_iid_iid, colors["iid_iid"])
    _annotate(4.5, mean_iid_cr, rel_iid_cr, colors["iid_cr"])

    # Formatting
    ax.set_xticks([1, 2, 3.5, 4.5])
    ax.set_xticklabels([
        r"Chiang $K^2$" + "\nIID SE",
        r"Chiang $K^2$" + "\nCR SE",
        "As-IID\nIID SE",
        "As-IID\nCR SE",
    ], fontsize=10)

    ax.set_ylabel("Model SE (dots) vs EmpSE (dashed)", fontsize=11)
    ax.set_title(
        "SE Calibration: IID vs Cluster-Robust Standard Errors\n"
        r"Chiang PLIV DGP ($N$=$M$=25, $p$=100, $K$=2)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0.3, 5.2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    results_csv = _PROJECT_ROOT / "replications" / "chiang" / "results" / "chiang_replication.csv"
    plots_dir = _PROJECT_ROOT / "replications" / "chiang" / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_se_calibration_dual(results_csv, plots_dir / "chiang_se_calibration_dual.png")
