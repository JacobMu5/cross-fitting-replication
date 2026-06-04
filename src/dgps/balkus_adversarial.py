"""Adversarial DGP with bounded clusters for Balkus et al. (2026) Corollary 2.

Satisfies all three assumptions of Corollary 2:
  1. Two-way CLT with r_n = sqrt(n) — bounded cluster sizes ensure this.
  2. Bounded cluster sizes — each row appears exactly d times, each column
     appears approximately d times (d-regular bipartite assignment).
  3. Nuisance regularity — standard learners converge on observed covariates.

Key structural change vs balkus_original:
  Shared latent confounder U_i (row) and V_j (col) drive covariates,
  propensity, outcome, AND treatment effect simultaneously. Cluster proxies
  (noisy U, V) are included as learner covariates so that a flexible model
  can exploit cluster identity — the leakage channel as-IID cannot block.

Design choices:
  - Sparse bipartite graph (d-regular) instead of full N*M grid
  - Propensity /5.0 compression matching Balkus
  - Gaussian noise matching Balkus
  - Cluster proxies: row_proxy = U_i + noise, col_proxy = V_j + noise

PATE ~ 1.0 by construction: E[sin(X)] = 0, E[X2] = 0, E[U*V] = 0.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import sigmoid


def _sparse_bipartite(
    N: int, d: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a d-regular bipartite graph via union of d random permutations.

    Both row and column degrees are exactly d, satisfying Corollary 2's
    bounded-cluster condition on both dimensions. Requires N=M.

    Args:
        N: Number of row/column clusters (must be equal).
        d: Degree — each row and each column appears exactly d times.
        rng: NumPy random generator.

    Returns:
        (row_ids, col_ids) arrays of length N*d.
    """
    row_ids, col_ids = [], []
    for _ in range(d):
        perm = rng.permutation(N)
        row_ids.append(np.arange(N))
        col_ids.append(perm)
    return np.concatenate(row_ids), np.concatenate(col_ids)


def generate_balkus_adversarial(
    N: int = 100, d: int = 5,
    sigma_Y: float = 0.5, proxy_mode: str = "noisy",
    proxy_noise: float = 0.5, seed: int = None,
) -> Tuple[pd.DataFrame, float]:
    """Generate one realization of the bounded-cluster adversarial DGP.

    Args:
        N: Number of row/column clusters (N=M enforced for d-regular graph).
        d: Bipartite degree (each row and column appears exactly d times).
        sigma_Y: Outcome noise standard deviation.
        proxy_mode: 'none' (X1-X5 only), 'noisy' (continuous proxies),
            or 'id' (row/column IDs as features — most adversarial).
        proxy_noise: Noise std for 'noisy' mode (higher = less leakage).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with covariates, treatment, outcome, oracles; PATE).
    """
    rng = np.random.default_rng(seed)

    # Shared latent confounders
    U = rng.normal(0, 1.5, N)
    V = rng.normal(0, 1.5, N)

    # Sparse bipartite assignment — bounded cluster sizes on both sides
    row_idx, col_idx = _sparse_bipartite(N, d, rng)
    n_obs = len(row_idx)

    # Covariates — U, V flow into X1
    X1 = U[row_idx] + V[col_idx] + rng.normal(0, 0.8, n_obs)
    X2 = rng.normal(0, 1, n_obs)
    X3 = np.sin(X1) + 0.3 * X2 + rng.normal(0, 0.5, n_obs)
    X4 = (X1 > 0).astype(float) * X2 + rng.normal(0, 0.5, n_obs)
    X5 = rng.normal(1, 2, n_obs)

    # Propensity depends ONLY on observed X — no unmeasured confounding
    lin_ps = (-0.4 + 0.8 * X1 - 0.7 * X2**2 + 0.5 * np.sin(X3)
              + 0.4 * X1 * X2 - 0.5 * X4 * X5)
    pA = sigmoid(lin_ps / 5.0)
    A = rng.binomial(1, pA)

    # Treatment effect — heterogeneous, depends on shared confounder
    tau_ij = (1.0 + 0.3 * np.sin(X1) - 0.2 * X2
              + 0.15 * U[row_idx] * V[col_idx])

    # Baseline outcome — nonlinear, driven by shared U, V through X1
    mu0 = (0.5 * X1**2 - 0.4 * X2 + 0.3 * np.sin(X1 * X2)
           + 0.5 * X3 - 0.2 * X4 * X5)

    Y = mu0 + A.astype(float) * tau_ij + rng.normal(0, sigma_Y, n_obs)
    ATE_true = float(np.mean(tau_ij))

    data = {
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
        "mu0_true": mu0,
        "pA_true": pA,
    }

    # Proxy variants — controls how much cluster info the learner sees
    if proxy_mode == "noisy":
        data["X6"] = U[row_idx] + rng.normal(0, proxy_noise, n_obs)
        data["X7"] = V[col_idx] + rng.normal(0, proxy_noise, n_obs)
    elif proxy_mode == "id":
        # Normalize to [0, 1] so tree-based learners can use them
        data["X6"] = row_idx / N
        data["X7"] = col_idx / N

    df = pd.DataFrame(data)
    return df, ATE_true

