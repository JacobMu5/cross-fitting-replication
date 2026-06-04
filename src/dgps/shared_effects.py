"""Shared-effects DGPs that remove Balkus's independence safety net."""

import numpy as np
import pandas as pd
from typing import Tuple

from src.utils import sigmoid


def generate_shared_effects(
    N: int = 20, M: int = 20, seed: int = None
) -> Tuple[pd.DataFrame, float]:
    """Shared cluster effects DGP — a single gamma drives L, A, and Y.

    Removes Balkus's three safety nets: independent cluster effects,
    propensity compression, and linear outcome model.

    Args:
        N: Number of row clusters.
        M: Number of column clusters.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with L1–L5, A, Y, cluster IDs; tau).
    """
    rng = np.random.default_rng(seed)

    gamma = rng.normal(0, 1.5, N)
    nu = rng.normal(0, 1.5, M)

    row_idx = np.repeat(np.arange(N), M)
    col_idx = np.tile(np.arange(M), N)
    n_obs = N * M

    L1 = rng.normal(gamma[row_idx] + nu[col_idx], 0.8, n_obs)
    L2 = rng.normal(0, 1.0, n_obs)
    L3 = rng.normal(np.sin(L1) + 0.3 * L2, np.sqrt(0.5), n_obs)
    L4 = rng.normal((L1 > 0).astype(float) * L2, np.sqrt(0.5), n_obs)
    L5 = rng.normal(1.0, np.sqrt(2.0), n_obs)

    eta_A = -0.3 + 0.8 * L1 - 0.4 * L2**2 + 0.3 * np.sin(L3)
    pA = sigmoid(eta_A)
    pA = np.clip(pA, 0.03, 0.97)
    A = rng.binomial(1, pA)

    mu0 = (0.5 * L1**2 - 0.4 * L2 + 0.3 * np.sin(L1 * L2)
           + 0.5 * L3 - 0.2 * L4 * L5)
    tau = 1.0
    Y = mu0 + A.astype(float) * tau + rng.normal(0, 0.7, n_obs)

    df = pd.DataFrame({
        "row_id": row_idx,
        "col_id": col_idx,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5,
        "A": A.astype(float),
        "Y": Y,
        "tau_true": tau,
        "mu0_true": mu0,
        "pA_true": pA,
    })

    return df, tau


def generate_shared_effects_hard(
    N: int = 20, M: int = 20, seed: int = None
) -> Tuple[pd.DataFrame, float]:
    """Hardened shared-effects DGP with extreme cluster-level confounding.

    Tests that BOTH As-IID and Chiang face structural failure boundaries in
    high-contrast environments: As-IID fails via cluster-level overfitting,
    Chiang fails because it cannot extrapolate extreme unobserved confounders.

    Args:
        N: Number of row clusters.
        M: Number of column clusters.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with L1–L5, A, Y, cluster IDs; tau).
    """
    rng = np.random.default_rng(seed)

    gamma = rng.normal(0, 3.0, N)
    nu = rng.normal(0, 3.0, M)

    row_idx = np.repeat(np.arange(N), M)
    col_idx = np.tile(np.arange(M), N)
    n_obs = N * M

    L1 = rng.normal(gamma[row_idx] + nu[col_idx], 0.1, n_obs)
    L2 = rng.normal(0, 1.0, n_obs)
    L3 = rng.normal(np.sin(L1) + 0.3 * L2, 0.5, n_obs)
    L4 = rng.normal((L1 > 0).astype(float) * L2, 0.5, n_obs)
    L5 = rng.normal(1.0, 1.0, n_obs)

    eta_A = -0.5 + 1.2 * L1 - 0.5 * L2**2 + 0.5 * np.sin(L3)
    pA = sigmoid(eta_A)
    pA = np.clip(pA, 0.01, 0.99)
    A = rng.binomial(1, pA)

    mu0 = (0.8 * L1**2 - 0.5 * L2 + 0.5 * np.sin(L1 * L2)
           + 0.8 * L3 - 0.5 * L4 * L5)
    tau = 1.0
    Y = mu0 + A.astype(float) * tau + rng.normal(0, 1.0, n_obs)

    df = pd.DataFrame({
        "row_id": row_idx,
        "col_id": col_idx,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5,
        "A": A.astype(float),
        "Y": Y,
        "tau_true": tau,
        "mu0_true": mu0,
        "pA_true": pA,
    })

    return df, tau
