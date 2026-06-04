"""Generate LaTeX table snippets from CSV results.

Usage:
    python -m report.generate_tables

Reads simulation results and writes .tex files to report/tables/.
The main .tex document includes them via \\input{report/tables/...}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation import compute_summary

TABLES_DIR = Path(__file__).parent / "tables"
BALKUS_RESULTS = _PROJECT_ROOT / "replications" / "balkus" / "results"
CHIANG_RESULTS = _PROJECT_ROOT / "replications" / "chiang" / "results"

LEARNER_LABELS = {"earth_exact": "MARS", "deep_rf": "Deep RF", "lasso": "LASSO"}
STRAT_ORDER = ["no_cf", "as_iid", "chiang"]
STRAT_LABELS = {"no_cf": r"no\_cf", "as_iid": r"as\_iid", "chiang": "chiang"}


def _fmt(val: float, fmt: str = ".2f") -> str:
    """Format a float, using bold for extreme RelSE values."""
    return f"{val:{fmt}}"


def _sign(val: float, fmt: str = ".3f") -> str:
    """Format with explicit sign."""
    return f"${'+' if val >= 0 else '-'}${abs(val):{fmt}}"


def _pct(val: float, decimals: int = 1) -> str:
    """Format a fraction as a LaTeX percentage (escapes %)."""
    return f"{val * 100:.{decimals}f}\\%"


def generate_relse_table() -> None:
    """Table: Mean RelSE by learner × strategy (Balkus DGPs)."""
    csv_path = BALKUS_RESULTS / "balkus_results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df = df[df["dgp"] != "balkus_hard"].copy()

    has_cr = df["se_cr"].notna() & df["covers_cr"].notna()
    sub = df[has_cr].copy()
    if sub.empty:
        print("  SKIP: no cluster-robust SE data")
        return

    if "n" not in sub.columns:
        sub["n"] = sub["N"] * sub["M"]

    group_cols = ["dgp", "learner", "strategy", "n"]
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

    # Aggregate by learner × strategy
    agg = merged.groupby(["learner", "strategy"]).agg(
        relse_iid=("rel_se_iid", "mean"),
        relse_cr=("rel_se_cr", "mean"),
        cov_cr=("cov_cr", "mean"),
    ).reset_index()

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (r"\textbf{Learner} & \textbf{Strategy} & "
         r"\textbf{RelSE (IID)} & \textbf{RelSE (CR)} & "
         r"\textbf{$\Delta$} & \textbf{Cov.\ CR} \\"),
        r"\midrule",
    ]

    for i, learner in enumerate(["earth_exact", "deep_rf", "lasso"]):
        if i > 0:
            lines.append(r"\midrule")
        label = LEARNER_LABELS[learner]
        rows_for_learner = agg[agg["learner"] == learner].set_index("strategy")

        for j, strat in enumerate(STRAT_ORDER):
            if strat not in rows_for_learner.index:
                continue
            row = rows_for_learner.loc[strat]
            delta = row["relse_cr"] - row["relse_iid"]

            # Bold the better RelSE (closer to 1.0)
            iid_str = _fmt(row["relse_iid"])
            cr_str = _fmt(row["relse_cr"])
            dist_iid = abs(row["relse_iid"] - 1.0)
            dist_cr = abs(row["relse_cr"] - 1.0)
            if dist_cr <= dist_iid:
                cr_str = r"\textbf{" + cr_str + "}"
            else:
                iid_str = r"\textbf{" + iid_str + "}"

            learner_col = (r"\multirow{3}{*}{" + label + "}")  if j == 0 else ""
            lines.append(
                f"  {learner_col} & {STRAT_LABELS[strat]} & "
                f"{iid_str} & {cr_str} & "
                f"${'+' if delta >= 0 else '-'}${abs(delta):.2f} & "
                f"{_pct(row['cov_cr'])} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABLES_DIR / "relse_balkus.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out}")


def generate_chiang_table() -> None:
    """Table: Chiang PLIV results from custom implementation."""
    csv_path = CHIANG_RESULTS / "chiang_summary_metrics.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Parse scenario string to extract (N,M) and dim
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (r"\textbf{Scenario} & \textbf{Strategy} & "
         r"\textbf{Bias} & \textbf{EmpSE} & "
         r"\textbf{RMSE} & \textbf{Cov.} \\"),
        r"\midrule",
    ]

    # Group by scenario
    scenarios = df["scenario"].unique()
    for i, scenario in enumerate(scenarios):
        if i > 0:
            lines.append(r"\midrule")
        sub = df[df["scenario"] == scenario].copy()
        # Clean up scenario name for LaTeX
        label_clean = scenario.replace("_", r"\_").replace(",", ", ")

        for j, (_, row) in enumerate(sub.iterrows()):
            strat = row["strategy"]
            strat_label = strat.replace("_", r"\_")
            scenario_col = label_clean if j == 0 else ""

            lines.append(
                f"  {scenario_col} & {strat_label} & "
                f"{_sign(row['bias'])} & "
                f"{row['emp_se']:.3f} & "
                f"{row['rmse']:.3f} & "
                f"{_pct(row['coverage'])} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABLES_DIR / "chiang_pliv.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out}")


def generate_doubleml_table() -> None:
    """Table: DoubleML head-to-head (if comparison PNG exists, the data is hardcoded
    in the validation script — read from the analysis output)."""
    # The DoubleML comparison data lives in the validation script output.
    # Check if a CSV was saved alongside the plot.
    csv_path = CHIANG_RESULTS / "doubleml_summary.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found (run doubleml_validation.py first)")
        return

    df = pd.read_csv(csv_path)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (r"\textbf{Strategy} & \textbf{Bias} & \textbf{EmpSE} & "
         r"\textbf{ModSE} & \textbf{RelSE} & \textbf{RMSE} & \textbf{Cov.} \\"),
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"  {row['strategy']} & "
            f"{_sign(row['bias'])} & "
            f"{row['emp_se']:.3f} & "
            f"{row['mod_se']:.3f} & "
            f"{row['rel_se']:.2f} & "
            f"{row['rmse']:.3f} & "
            f"{_pct(row['coverage'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABLES_DIR / "doubleml_comparison.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out}")


def generate_decoupling_table() -> None:
    """Table: Coverage under each (strategy × SE type) combination.

    Direct evidence that as-IID + CR SE is the optimal combination.
    """
    csv_path = BALKUS_RESULTS / "balkus_results.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df = df[df["dgp"] != "balkus_hard"].copy()

    has_cr = df["se_cr"].notna() & df["covers_cr"].notna()
    sub = df[has_cr].copy()
    if sub.empty:
        print("  SKIP: no cluster-robust SE data")
        return

    if "n" not in sub.columns:
        sub["n"] = sub["N"] * sub["M"]

    group_cols = ["dgp", "learner", "strategy", "n"]
    summary_iid = compute_summary(sub, group_cols=group_cols,
                                  se_col="se_iid", covers_col="covers_iid")
    summary_cr = compute_summary(sub, group_cols=group_cols,
                                 se_col="se_cr", covers_col="covers_cr")

    merged = summary_iid[group_cols + ["coverage", "rel_se"]].rename(
        columns={"coverage": "cov_iid", "rel_se": "relse_iid"}
    ).merge(
        summary_cr[group_cols + ["coverage", "rel_se"]].rename(
            columns={"coverage": "cov_cr", "rel_se": "relse_cr"}
        ),
        on=group_cols,
    )

    # Aggregate by learner × strategy
    agg = merged.groupby(["learner", "strategy"]).agg(
        cov_iid=("cov_iid", "mean"),
        cov_cr=("cov_cr", "mean"),
        relse_iid=("relse_iid", "mean"),
        relse_cr=("relse_cr", "mean"),
    ).reset_index()

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (r"\textbf{Learner} & \textbf{Strategy} & "
         r"\textbf{Cov.\ (IID SE)} & \textbf{Cov.\ (CR SE)} & "
         r"\textbf{RelSE (IID)} & \textbf{RelSE (CR)} \\"),
        r"\midrule",
    ]

    for i, learner in enumerate(["earth_exact", "deep_rf", "lasso"]):
        if i > 0:
            lines.append(r"\midrule")
        label = LEARNER_LABELS[learner]
        rows = agg[agg["learner"] == learner].set_index("strategy")

        for j, strat in enumerate(STRAT_ORDER):
            if strat not in rows.index:
                continue
            row = rows.loc[strat]
            learner_col = (r"\multirow{3}{*}{" + label + "}") if j == 0 else ""

            # Bold the best coverage per learner (closest to 95%)
            cov_iid_str = _pct(row['cov_iid'])
            cov_cr_str = _pct(row['cov_cr'])
            if abs(row["cov_cr"] - 0.95) < abs(row["cov_iid"] - 0.95):
                cov_cr_str = r"\textbf{" + cov_cr_str + "}"
            else:
                cov_iid_str = r"\textbf{" + cov_iid_str + "}"

            lines.append(
                f"  {learner_col} & {STRAT_LABELS[strat]} & "
                f"{cov_iid_str} & {cov_cr_str} & "
                f"{row['relse_iid']:.2f} & {row['relse_cr']:.2f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABLES_DIR / "decoupling_proof.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {out}")


def main() -> None:
    """Generate all LaTeX table snippets."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating LaTeX tables from CSV results...")

    generate_relse_table()
    generate_chiang_table()
    generate_doubleml_table()
    generate_decoupling_table()

    print(f"\nAll tables written to {TABLES_DIR}/")
    print("Include in .tex with: \\input{report/tables/relse_balkus}")


if __name__ == "__main__":
    main()
