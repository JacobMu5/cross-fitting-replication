"""Full-grid DGP where chiang should dominate as-IID cross-fitting.

Violates Corollary 2's bounded-cluster assumption: the full N*M grid
gives each row M observations and each column N — unbounded as N,M grow.

Design rationale for chiang dominance:
  1. Strong direct U,V effect on outcome (terms NOT mediated by X1)
     → learner MUST recover U,V to predict well
  2. Noisy cluster proxies as covariates → partial U,V information
  3. In as-IID: training fold shares clusters with holdout →
     RF averages proxy across M-1 same-row training obs → denoises U →
     nuisance prediction is "too good" → EP bias doesn't vanish
  4. In chiang: no shared clusters → RF can't denoise proxy for
     holdout clusters → noisier prediction, but EP bias vanishes
  5. Large enough n=N*M that chiang's K²-fold data loss isn't fatal

Propensity depends ONLY on observed X1-X5 → unconfoundedness holds.
PATE ~ 1.0 by construction.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import sigmoid


def generate_balkus_fullgrid(
    N: int = 40, M: int = 40, sigma_Y: float = 0.5,
    proxy_noise: float = 0.3, seed: int = None,
) -> Tuple[pd.DataFrame, float]:
    """Generate one realization of the full-grid DGP.

    Args:
        N: Number of row clusters.
        M: Number of column clusters.
        sigma_Y: Outcome noise standard deviation.
        proxy_noise: Noise std for cluster proxies (lower = more leakage).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with covariates, treatment, outcome, oracles; PATE).
    """
    rng = np.random.default_rng(seed)

    # Strong shared latent confounders
    U = rng.normal(0, 2.0, N)
    V = rng.normal(0, 2.0, M)

    # Full grid — each row has M obs, each column has N obs (UNBOUNDED)
    row_idx = np.repeat(np.arange(N), M)
    col_idx = np.tile(np.arange(M), N)
    n_obs = N * M

    # Covariates — U, V partially visible through X1
    X1 = U[row_idx] + V[col_idx] + rng.normal(0, 0.5, n_obs)
    X2 = rng.normal(0, 1, n_obs)
    X3 = np.sin(X1) + 0.3 * X2 + rng.normal(0, 0.5, n_obs)
    X4 = (X1 > 0).astype(float) * X2 + rng.normal(0, 0.5, n_obs)
    X5 = rng.normal(1, 2, n_obs)

    # Cluster proxies — noisy U, V. Low noise = strong signal for RF
    X6 = U[row_idx] + rng.normal(0, proxy_noise, n_obs)
    X7 = V[col_idx] + rng.normal(0, proxy_noise, n_obs)

    # Propensity depends ONLY on observed X — no unmeasured confounding
    lin_ps = (-0.4 + 0.8 * X1 - 0.7 * X2**2 + 0.5 * np.sin(X3)
              + 0.4 * X1 * X2 - 0.5 * X4 * X5)
    pA = sigmoid(lin_ps / 5.0)
    A = rng.binomial(1, pA)

    # Treatment effect — heterogeneous, U*V interaction (stronger than bounded)
    tau_ij = (1.0 + 0.3 * np.sin(X1) - 0.2 * X2
              + 0.4 * U[row_idx] * V[col_idx])

    # Baseline outcome — strong DIRECT cluster effects not mediated by X1.
    # The U*sin(V) and |U|*X2 terms require cluster identity knowledge.
    # In as-IID, the RF denoises proxy from M-1 same-row training obs.
    # In chiang, the RF cannot — it only sees the noisy proxy.
    mu0 = (0.5 * X1**2 - 0.4 * X2 + 0.3 * np.sin(X1 * X2)
           + 0.5 * X3 - 0.2 * X4 * X5
           + 0.8 * U[row_idx] * np.sin(V[col_idx])
           + 0.4 * np.abs(U[row_idx]) * X2)

    Y = mu0 + A.astype(float) * tau_ij + rng.normal(0, sigma_Y, n_obs)
    ATE_true = float(np.mean(tau_ij))

    df = pd.DataFrame({
        "row_id": row_idx,
        "col_id": col_idx,
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
        "X6": X6,
        "X7": X7,
        "A": A.astype(float),
        "Y": Y,
        "tau_true": tau_ij,
        "mu0_true": mu0,
        "pA_true": pA,
    })

    return df, ATE_true
