"""Extreme bounded-cluster DGP for Balkus et al. (2026) stress test.

Same d-regular bipartite structure as balkus_adversarial, but with
amplified cluster effects to stress-test the finite-sample constant
in Corollary 2's empirical process bound:
  - U, V scale 5.0 (vs 1.5 in balkus_adversarial)
  - Proxy noise 0.1 (vs 0.5)
  - Direct U*sin(V) and |U|*X2 terms in outcome
  - 0.4*U*V interaction in treatment effect

Bounded-cluster condition is satisfied (d-regular graph), but the
leakage channel is deliberately amplified. Designed to answer: does
Chiang's bias advantage persist as N grows, and is it worth the
variance cost?
"""

import numpy as np
import pandas as pd
from typing import Tuple

from src.dgps.balkus_adversarial import _sparse_bipartite
from src.utils import sigmoid


def generate_balkus_extreme(
    N: int = 100, d: int = 20, sigma_Y: float = 0.5,
    seed: int = None,
) -> Tuple[pd.DataFrame, float]:
    """Generate one realization of the extreme bounded-cluster DGP.

    Args:
        N: Number of row/column clusters (N=M enforced for d-regular graph).
        d: Bipartite degree (each row and column appears exactly d times).
        sigma_Y: Outcome noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame with covariates, treatment, outcome, oracles;
        finite-sample SATE for this realization).
    """
    rng = np.random.default_rng(seed)

    # Amplified shared confounders — 3.3x stronger than balkus_adversarial
    U = rng.normal(0, 5.0, N)
    V = rng.normal(0, 5.0, N)

    row_idx, col_idx = _sparse_bipartite(N, d, rng)
    n = len(row_idx)

    # Covariates — tighter noise (0.5 vs 0.8) amplifies U,V signal
    X1 = U[row_idx] + V[col_idx] + rng.normal(0, 0.5, n)
    X2 = rng.normal(0, 1, n)
    X3 = np.sin(X1) + 0.3 * X2 + rng.normal(0, 0.5, n)
    X4 = (X1 > 0).astype(float) * X2 + rng.normal(0, 0.5, n)
    X5 = rng.normal(1, 2, n)

    # Near-perfect cluster proxies — RF can recover U from ~d/2 training obs
    X6 = U[row_idx] + rng.normal(0, 0.1, n)
    X7 = V[col_idx] + rng.normal(0, 0.1, n)

    # Propensity ONLY on observed X — no unmeasured confounding
    lin_ps = (-0.4 + 0.8 * X1 - 0.7 * X2**2 + 0.5 * np.sin(X3)
              + 0.4 * X1 * X2 - 0.5 * X4 * X5)
    pA = sigmoid(lin_ps / 5.0)
    A = rng.binomial(1, pA)

    # Treatment effect with strong U*V interaction
    tau = (1.0 + 0.3 * np.sin(X1) - 0.2 * X2
           + 0.4 * U[row_idx] * V[col_idx])

    # Outcome — strong DIRECT cluster dependence
    mu0 = (0.5 * X1**2 - 0.4 * X2 + 0.3 * np.sin(X1 * X2)
           + 0.5 * X3 - 0.2 * X4 * X5
           + 0.8 * U[row_idx] * np.sin(V[col_idx])
           + 0.4 * np.abs(U[row_idx]) * X2)

    Y = mu0 + A * tau + rng.normal(0, sigma_Y, n)
    ate = float(np.mean(tau))

    df = pd.DataFrame({
        "row_id": row_idx, "col_id": col_idx,
        "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
        "X6": X6, "X7": X7,
        "A": A.astype(float), "Y": Y,
        "tau_true": tau, "mu0_true": mu0, "pA_true": pA,
    })
    return df, ate
