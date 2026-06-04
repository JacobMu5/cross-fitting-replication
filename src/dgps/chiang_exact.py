"""Chiang et al. (2021, Section 4.1) exact partially linear IV DGP.

Reference implementation:
  https://github.com/salbalkus/cross-fitting-dependent-data/blob/main/R/dgps.R  (dgp_cluster)
  Chiang et al. (2021, JBES), arXiv:1909.03489, Section 5.1.
"""

import numpy as np
from scipy.linalg import toeplitz, cholesky
from typing import Any, Dict, Tuple


def generate_chiang_exact(
    N: int = 20, M: int = 20, seed: int = None,
    dim_x: int = 100,
    theta_0: float = 1.0,
    pi_10: float = 1.0,
    s_x: float = 0.25,
    s_eu: float = 0.25,
    w1: float = 0.25, w2: float = 0.25,
    decaying_coef: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """Generate one draw from the Chiang et al. (2021) Section 4.1 DGP.

    Args:
        N: Number of row clusters (products).
        M: Number of column clusters (markets).
        seed: Random seed.
        dim_x: Dimension of covariates X.
        theta_0: True structural parameter (target of inference).
        pi_10: First-stage coefficient on instrument Z.
        s_x: Autocovariance for Toeplitz covariance, Sigma_X[p,q] = s_x^|p-q|.
        s_eu: Endogeneity correlation between eps and upsilon shocks.
        w1, w2: Clustering weights for row and column dimensions.
        decaying_coef: If True, use paper's (0.5, 0.5², ..., 0.5^p); if False,
            use Balkus R constant rep(0.5, p).

    Returns:
        Tuple of (data dict with Y, D, X, Z, row_id, col_id; theta_0).
    """
    rng = np.random.default_rng(seed)
    n_obs = N * M

    # Coefficient vectors — Chiang et al. (2021, Section 4.1, line 704)
    if decaying_coef:
        # Paper: ζ₀ = π₂₀ = ξ₀ = (0.5, 0.5², ..., 0.5^p)'
        coef_vec = 0.5 ** np.arange(1, dim_x + 1)
    else:
        # Balkus R: rep(0.5, p) — constant vector
        coef_vec = np.full(dim_x, 0.5)

    zeta_0 = coef_vec
    pi_20 = coef_vec
    xi_0 = coef_vec

    # Toeplitz covariance for X — Chiang et al. (2021, Section 4.1)
    first_row = s_x ** np.arange(dim_x, dtype=np.float64)
    Sigma_X = toeplitz(first_row)
    L_X = cholesky(Sigma_X, lower=True)

    # Endogeneity covariance for (eps, upsilon)
    Sigma_eu = np.array([[1.0, s_eu], [s_eu, 1.0]])
    L_eu = cholesky(Sigma_eu, lower=True)

    # Draw cluster-level and idiosyncratic shocks
    def draw_X_shocks(shape_leading: tuple) -> np.ndarray:
        """Draw N(0, Sigma_X) samples with given leading dimension."""
        raw = rng.standard_normal((*shape_leading, dim_x))
        return raw @ L_X.T

    aX_ij = draw_X_shocks((N, M))
    aX_i = draw_X_shocks((N,))
    aX_j = draw_X_shocks((M,))

    def draw_eu_shocks(shape_leading: tuple) -> np.ndarray:
        """Draw BivariateNormal(0, Sigma_eu) samples."""
        raw = rng.standard_normal((*shape_leading, 2))
        return raw @ L_eu.T

    aeu_ij = draw_eu_shocks((N, M))
    aeu_i = draw_eu_shocks((N,))
    aeu_j = draw_eu_shocks((M,))

    aV_ij = rng.standard_normal((N, M))
    aV_i = rng.standard_normal(N)
    aV_j = rng.standard_normal(M)

    # Construct two-way clustered variables
    w0 = 1.0 - w1 - w2

    X_full = (w0 * aX_ij
              + w1 * aX_i[:, np.newaxis, :]
              + w2 * aX_j[np.newaxis, :, :])

    eu_full = (w0 * aeu_ij
               + w1 * aeu_i[:, np.newaxis, :]
               + w2 * aeu_j[np.newaxis, :, :])
    eps_full = eu_full[:, :, 0]
    upsilon_full = eu_full[:, :, 1]

    V_full = (w0 * aV_ij
              + w1 * aV_i[:, np.newaxis]
              + w2 * aV_j[np.newaxis, :])

    # Structural equations — Chiang et al. (2021, Section 4.1)
    Z_full = X_full @ xi_0 + V_full
    D_full = Z_full * pi_10 + X_full @ pi_20 + upsilon_full
    Y_full = D_full * theta_0 + X_full @ zeta_0 + eps_full

    # Flatten to observation-level arrays
    row_ids = np.repeat(np.arange(N), M)
    col_ids = np.tile(np.arange(M), N)

    Y = Y_full.reshape(-1)
    D = D_full.reshape(-1)
    Z = Z_full.reshape(-1)
    X = X_full.reshape(n_obs, dim_x)

    data = {
        "Y": Y,
        "D": D,
        "X": X,
        "Z": Z,
        "row_id": row_ids,
        "col_id": col_ids,
    }

    return data, theta_0
