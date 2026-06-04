"""DoubleML head-to-head: Chiang K²-fold vs as-IID on the official DGP.

Standalone validation using the DoubleML library's own implementation of
Chiang et al. (2021, Section 4.1) DGP with decaying coefficients (ζ₀)_j = 0.5^j.

This script is NOT part of the Morozov orchestrator pipeline because it uses an
external library with its own cross-fitting logic. It complements the custom
implementation results from replications/chiang/main.py.

Key insight: DoubleML bundles cross-fitting strategy with variance estimation.
Setting cluster_cols=None strips BOTH cluster-aware splitting AND cluster-robust
variance, making it impossible to test Balkus's (2026) recommended combination
(as-IID splitting + cluster-robust SE) through their API alone.

Usage:
    python -m replications.chiang.doubleml_validation
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.base import clone

from doubleml import DoubleMLPLIV, DoubleMLData
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SIM = 100
FIRST_SEED = 7691
N, M, DIM_X = 25, 25, 100
K = 3
THETA_TRUE = 1.0


def run_one_rep(seed: int) -> dict:
    """Run a single replication comparing both DoubleML strategies.

    Args:
        seed: Random seed for this replication.

    Returns:
        Dict with theta and SE from both strategies.
    """
    np.random.seed(seed)

    data_cluster = make_pliv_multiway_cluster_CKMS2021(N, M, DIM_X)

    # Strategy 1: Chiang K²-fold (cluster-aware splitting + cluster-robust SE)
    learner = LassoCV(max_iter=5000)
    dml_chiang = DoubleMLPLIV(
        data_cluster,
        ml_l=clone(learner), ml_m=clone(learner), ml_r=clone(learner),
        n_folds=K,
    )
    dml_chiang.fit()

    theta_chiang = dml_chiang.coef[0]
    se_chiang = dml_chiang.se[0]

    # Strategy 2: As-IID (standard KFold, K²=9 folds to match total)
    df = data_cluster.data.copy()
    data_iid = DoubleMLData(
        df.drop(columns=["cluster_var_i", "cluster_var_j"]),
        y_col="Y", d_cols="D", x_cols=[f"X{j}" for j in range(1, DIM_X + 1)],
        z_cols="Z",
    )
    dml_iid = DoubleMLPLIV(
        data_iid,
        ml_l=clone(learner), ml_m=clone(learner), ml_r=clone(learner),
        n_folds=K ** 2,
    )
    dml_iid.fit()

    theta_iid = dml_iid.coef[0]
    se_iid = dml_iid.se[0]

    return {
        "seed": seed,
        "theta_chiang": theta_chiang,
        "se_chiang": se_chiang,
        "bias_chiang": theta_chiang - THETA_TRUE,
        "theta_iid": theta_iid,
        "se_iid": se_iid,
        "bias_iid": theta_iid - THETA_TRUE,
    }


if __name__ == "__main__":
    from joblib import Parallel, delayed

    n_jobs = os.cpu_count()
    seeds = [FIRST_SEED + i for i in range(N_SIM)]

    print(f"Running {N_SIM} reps on DoubleML Chiang DGP (N={N}, M={M}, dim={DIM_X}, K={K})")
    print(f"True theta_0 = {THETA_TRUE}, using {n_jobs} cores\n")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_rep)(s) for s in seeds
    )
    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)

    # Print Morris et al. (2019) performance measures
    print(f"\n{'='*60}")
    print(f"{'DoubleML Head-to-Head Results':^60}")
    print(f"{'='*60}")

    for label, prefix in [("Chiang K^2-fold", "chiang"), ("As-IID", "iid")]:
        bias = df[f"bias_{prefix}"].mean()
        emp_se = df[f"theta_{prefix}"].std()
        mod_se = df[f"se_{prefix}"].mean()
        rmse = np.sqrt((df[f"bias_{prefix}"] ** 2).mean())
        rel_se = mod_se / emp_se if emp_se > 0 else float("nan")

        covers = (
            (df[f"theta_{prefix}"] - 1.96 * df[f"se_{prefix}"] <= THETA_TRUE)
            & (THETA_TRUE <= df[f"theta_{prefix}"] + 1.96 * df[f"se_{prefix}"])
        ).mean()

        print(f"\n  {label}:")
        print(f"    Bias:      {bias:+.4f}")
        print(f"    EmpSE:     {emp_se:.4f}")
        print(f"    ModSE:     {mod_se:.4f}")
        print(f"    RelSE:     {rel_se:.3f}")
        print(f"    RMSE:      {rmse:.4f}")
        print(f"    Coverage:  {covers * 100:.1f}%")

    out_path = RESULTS_DIR / "doubleml_head_to_head_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
