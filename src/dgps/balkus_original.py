"""Balkus et al. (2026) two-way clustered DGP (arXiv:2601.10899, Section 3.1).

Reference R implementation:
  https://github.com/salbalkus/cross-fitting-dependent-data/blob/main/R/hard_cluster.R
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import sigmoid


def generate_balkus_original(
    N: int = 20, M: int = 20, sigma_Y: float = 0.5, seed: int = None
) -> Tuple[pd.DataFrame, float]:
    """Generate one realization of the Balkus et al. (2026) two-way clustered DGP.

    Key structural properties that immunize this DGP against cross-fold leakage:
    independent cluster effects (a.X ⊥ a.m ⊥ a.0 ⊥ a.t), propensity
    compression (logit / 5.0), heterogeneous treatment effect tau(x).

    Args:
        N: Number of row clusters.
        M: Number of column clusters.
        sigma_Y: Outcome noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with X1–X5, A, Y, row/col IDs, oracles; ATE_true).
    """
    rng = np.random.default_rng(seed)

    # Independent cluster effects — Balkus et al. (2026, hard_cluster.R)
    a_X = rng.normal(0, 1, N)
    b_X = rng.normal(0, 1, M)
    a_m = rng.normal(0, 1, N)   # propensity
    b_m = rng.normal(0, 1, M)
    a_t = rng.normal(0, 1, N)   # heterogeneous treatment effect
    b_t = rng.normal(0, 1, M)
    a_0 = rng.normal(0, 1, N)   # baseline outcome
    b_0 = rng.normal(0, 1, M)

    # Vectorized grid setup
    row_idx = np.repeat(np.arange(N), M)
    col_idx = np.tile(np.arange(M), N)
    n_obs = N * M

    # Covariates — Balkus et al. (2026, hard_cluster.R, lines 23–27)
    X1 = a_X[row_idx] + b_X[col_idx] + rng.normal(0, 1, n_obs)
    X2 = rng.normal(0, 1, n_obs)
    X3 = np.sin(X1) + 0.3 * X2 + rng.normal(0, 0.5, n_obs)
    X4 = (X1 > 0).astype(float) * X2 + rng.normal(0, 0.5, n_obs)
    X5 = rng.normal(1, 2, n_obs)

    # Propensity: /5.0 compression — Balkus et al. (2026, hard_cluster.R, line 29–31)
    lin_ps = (-0.4 + 0.8 * X1 - 0.7 * X2**2 + 0.5 * np.sin(X3)
              + 0.4 * X1 * X2 - 0.5 * X4 * X5
              + 0.6 * a_m[row_idx] - 0.6 * b_m[col_idx])
    pA = sigmoid(lin_ps / 5.0)
    A = rng.binomial(1, pA)

    # Heterogeneous treatment effect — Balkus et al. (2026, hard_cluster.R, line 36)
    tau_ij = (1.0 + 0.5 * np.sin(X1) - 0.5 * np.sin(X2)
              + 0.3 * X3 * X4 + 0.4 * a_t[row_idx] * b_t[col_idx])

    # Baseline outcome — Balkus et al. (2026, hard_cluster.R, lines 40–41)
    Q0 = (2.0 + 0.5 * X1 - 0.4 * X2 + 0.3 * X3**2
          - 0.5 * np.sin(X4) + 0.4 * X1 * X2
          + 0.6 * a_0[row_idx] + 0.6 * b_0[col_idx])

    Y = Q0 + A.astype(float) * tau_ij + rng.normal(0, sigma_Y, n_obs)

    # ATE = E[tau(X)] — sample average of heterogeneous effects
    ATE_true = float(np.mean(tau_ij))

    df = pd.DataFrame({
        "row_id": row_idx,
        "col_id": col_idx,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
        "A": A.astype(float),
        "Y": Y,
        "tau_true": tau_ij,
        "mu0_true": Q0,
        "pA_true": pA,
    })

    return df, ATE_true
