"""Evaluation metrics following Morris, White & Crowther (2019, Statistics in Medicine)."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


def compute_summary(
    df: pd.DataFrame, group_cols: Optional[List[str]] = None,
    se_col: str = "se", covers_col: str = "covers",
) -> pd.DataFrame:
    """Compute Morris et al. (2019) performance measures from raw MC results.

    Metrics: Bias, EmpSE, ModSE, RelSE, MSE, RMSE, Coverage, CI Length,
    each with Monte Carlo standard errors (MCSEs).

    Args:
        df: Raw replication-level results with columns: bias, se, covers.
        group_cols: Columns to group by (default: dgp, learner, strategy).
        se_col: Column name for model-based SE (default: 'se'). Use 'se_cr'
            for cluster-robust Morris diagnostics.
        covers_col: Column name for coverage indicator (default: 'covers').
            Use 'covers_cr' for cluster-robust coverage.

    Returns:
        DataFrame with one row per group and all performance metrics.
    """
    if group_cols is None:
        group_cols = ["dgp", "learner", "strategy"]
        group_cols = [c for c in group_cols if c in df.columns]

    rows: List[Dict[str, Any]] = []
    for key, sub in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)

        n_reps = len(sub)
        biases = sub["bias"].values
        ses = sub[se_col].values

        # Bias: E[θ̂ − θ₀] — Morris et al. (2019, eq. 10)
        bias_mean = np.mean(biases)
        emp_se = np.std(biases, ddof=1)
        bias_mcse = emp_se / np.sqrt(n_reps)

        # EmpSE — Morris et al. (2019, eq. 12)
        emp_se_mcse = emp_se / np.sqrt(2 * (n_reps - 1))

        # ModSE: √E[SE²(θ̂)] — Morris et al. (2019, Section 5.4)
        # RMS form is correct: we need E[SE²] = EmpSE², not E[SE] = EmpSE
        mod_se = np.sqrt(np.mean(ses ** 2))
        mod_se_mcse = np.std(ses, ddof=1) / np.sqrt(n_reps)

        # RelSE: ModSE / EmpSE — Morris et al. (2019, Section 5.4)
        rel_se = mod_se / emp_se if emp_se > 0 else np.nan
        if emp_se > 0 and n_reps > 2:
            rel_se_mcse = rel_se * np.sqrt(
                (mod_se_mcse / mod_se) ** 2 + (emp_se_mcse / emp_se) ** 2
            ) if mod_se > 0 else np.nan
        else:
            rel_se_mcse = np.nan

        # MSE and RMSE
        mse = bias_mean ** 2 + emp_se ** 2
        rmse = np.sqrt(mse)

        # Coverage (binomial proportion)
        coverage = np.mean(sub[covers_col].values)
        coverage_mcse = np.sqrt(coverage * (1 - coverage) / n_reps)

        # Mean CI Length
        if "ci_lower" in sub.columns and "ci_upper" in sub.columns:
            ci_lengths = sub["ci_upper"].values - sub["ci_lower"].values
            mean_ci_length = np.mean(ci_lengths)
        else:
            mean_ci_length = 2 * 1.96 * mod_se

        # Noack et al. (2026) metrics
        se_noack_val = np.nan
        cov_noack_val = np.nan
        if "se_noack_cr" in sub.columns:
            se_noack_nonan = sub["se_noack_cr"].dropna().values
            if len(se_noack_nonan) > 0:
                se_noack_val = np.mean(se_noack_nonan)
        if "covers_noack_cr" in sub.columns:
            cov_noack_nonan = sub["covers_noack_cr"].dropna().values
            if len(cov_noack_nonan) > 0:
                cov_noack_val = np.mean(cov_noack_nonan)

        row = dict(zip(group_cols, key))
        row.update({
            "n_reps": n_reps,
            "bias": bias_mean,
            "bias_mcse": bias_mcse,
            "emp_se": emp_se,
            "emp_se_mcse": emp_se_mcse,
            "mod_se": mod_se,
            "mod_se_mcse": mod_se_mcse,
            "rel_se": rel_se,
            "rel_se_mcse": rel_se_mcse,
            "mse": mse,
            "rmse": rmse,
            "coverage": coverage,
            "coverage_mcse": coverage_mcse,
            "mean_ci_length": mean_ci_length,
            "se_noack_cr": se_noack_val,
            "cov_noack_cr": cov_noack_val,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary table with MCSEs to stdout."""
    summary = compute_summary(df)

    header = (
        f"{'DGP':>20s} | {'Learner':>10s} | {'Strategy':>8s} | "
        f"{'Bias':>10s} | {'EmpSE':>8s} | {'ModSE':>8s} | "
        f"{'RelSE':>7s} | {'Cov':>10s} | {'CI Len':>7s}"
    )
    print(header)
    print("-" * len(header))

    for _, row in summary.iterrows():
        marker = ""
        if row["coverage"] < 0.90:
            marker = " [!]"
        if row["rel_se"] < 0.85:
            marker += " [SE_low]"
        elif row["rel_se"] > 1.15:
            marker += " [SE_high]"

        bias_str = f"{row['bias']:+.4f} ({row['bias_mcse']:.4f})"
        cov_str = f"{row['coverage']:.1%} ({row['coverage_mcse']:.1%})"

        print(
            f"{row.get('dgp', ''):>20s} | "
            f"{row.get('learner', ''):>10s} | "
            f"{row.get('strategy', ''):>8s} | "
            f"{bias_str:>10s} | "
            f"{row['emp_se']:>8.4f} | "
            f"{row['mod_se']:>8.4f} | "
            f"{row['rel_se']:>7.3f} | "
            f"{cov_str:>10s} | "
            f"{row['mean_ci_length']:>7.4f}"
            f"{marker}"
        )

    n_reps_min = summary["n_reps"].min()
    print(f"\nn_reps = {n_reps_min}–{summary['n_reps'].max()}")
    print("RelSE = ModSE/EmpSE. Values < 1 -> SEs anti-conservative; "
          "> 1 -> SEs conservative.")
    print("MCSE = Monte Carlo Standard Error (in parentheses).")
    print("[!] = coverage < 90%.  [SE_low] = RelSE < 0.85.  "
          "[SE_high] = RelSE > 1.15.")
