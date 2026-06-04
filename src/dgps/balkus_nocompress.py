"""Balkus new_cluster.R DGP — no propensity compression.

Identical structure to balkus_original.py except:
  - plogis(lin_ps) instead of plogis(lin_ps / 5) → propensity spreads to [0.05, 0.95]
  - Extra term in tau: -0.4 * (X2 > 0) — matching new_cluster.R line 43

Reference R implementation:
  https://github.com/salbalkus/cross-fitting-dependent-data/blob/main/R/new_cluster.R
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import sigmoid


def generate_balkus_nocompress(
    N: int = 20, M: int = 20, sigma_Y: float = 0.5, seed: int = None
) -> Tuple[pd.DataFrame, float]:
    """Balkus DGP without propensity compression — matching new_cluster.R.

    Without /5 compression, propensity scores spread to extremes,
    making DR weights more sensitive to nuisance estimation errors.

    Args:
        N: Number of row clusters.
        M: Number of column clusters.
        sigma_Y: Outcome noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with X1–X5, A, Y, row/col IDs, oracles; ATE_true).
    """
    rng = np.random.default_rng(seed)

    a_X = rng.normal(0, 1, N)
    b_X = rng.normal(0, 1, M)
    a_m = rng.normal(0, 1, N)
    b_m = rng.normal(0, 1, M)
    a_t = rng.normal(0, 1, N)
    b_t = rng.normal(0, 1, M)
    a_0 = rng.normal(0, 1, N)
    b_0 = rng.normal(0, 1, M)

    row_idx = np.repeat(np.arange(N), M)
    col_idx = np.tile(np.arange(M), N)
    n_obs = N * M

    # Covariates — same as hard_cluster.R / balkus_original
    X1 = a_X[row_idx] + b_X[col_idx] + rng.normal(0, 1, n_obs)
    X2 = rng.normal(0, 1, n_obs)
    X3 = np.sin(X1) + 0.3 * X2 + rng.normal(0, 0.5, n_obs)
    X4 = (X1 > 0).astype(float) * X2 + rng.normal(0, 0.5, n_obs)
    X5 = rng.normal(1, 2, n_obs)

    # Propensity: NO compression — new_cluster.R line 38: plogis(lin_ps)
    lin_ps = (-0.4 + 0.8 * X1 - 0.7 * X2**2 + 0.5 * np.sin(X3)
              + 0.4 * X1 * X2 - 0.5 * X4 * X5
              + 0.6 * a_m[row_idx] - 0.6 * b_m[col_idx])
    pA = sigmoid(lin_ps)  # no /5 compression
    A = rng.binomial(1, pA)

    # Treatment effect — new_cluster.R line 43: extra -0.4*(X2 > 0) term
    tau_ij = (1.0 + 0.5 * np.sin(X1) - 0.5 * np.sin(X2)
              - 0.4 * (X2 > 0).astype(float)
              + 0.3 * X3 * X4 + 0.4 * a_t[row_idx] * b_t[col_idx])

    # Baseline outcome — same as hard_cluster.R
    Q0 = (2.0 + 0.5 * X1 - 0.4 * X2 + 0.3 * X3**2
          - 0.5 * np.sin(X4) + 0.4 * X1 * X2
          + 0.6 * a_0[row_idx] + 0.6 * b_0[col_idx])

    Y = Q0 + A.astype(float) * tau_ij + rng.normal(0, sigma_Y, n_obs)
    ATE_true = float(np.mean(tau_ij))

    df = pd.DataFrame({
        "row_id": row_idx,
        "col_id": col_idx,
        "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
        "A": A.astype(float),
        "Y": Y,
        "tau_true": tau_ij,
        "mu0_true": Q0,
        "pA_true": pA,
    })

    return df, ATE_true
