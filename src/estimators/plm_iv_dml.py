"""DML2 estimator for the partially linear IV model (Chiang et al. 2021, Algorithm 1).

Reference: Chiang, Kato, Ma & Sasaki (2021, JBES), arXiv:1909.03489.
Variance: Cameron, Gelbach & Miller (2011, JBES) two-way cluster-robust formula.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import Lasso, LassoCV, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor


def _make_lasso(seed: int, dim_x: int) -> LassoCV:
    """Lasso with cross-validated regularization (Chiang's default)."""
    return LassoCV(cv=3, max_iter=5000, random_state=seed, n_jobs=1)


def _make_ridge(seed: int, dim_x: int) -> Ridge:
    """Ridge regression (Chiang's alternative)."""
    return Ridge(alpha=1.0)


def _make_elasticnet(seed: int, dim_x: int) -> ElasticNet:
    """Elastic net (Chiang's alternative)."""
    return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                      random_state=seed)


def _make_rf(seed: int, dim_x: int) -> RandomForestRegressor:
    """Deep Random Forest — genuinely non-Donsker."""
    return RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        random_state=seed, n_jobs=1
    )


ML_METHODS = {
    "lasso": _make_lasso,
    "ridge": _make_ridge,
    "elasticnet": _make_elasticnet,
    "rf": _make_rf,
}


def _make_k2_partitions(
    N: int, M: int, K: int, rng: np.random.Generator
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Randomly partition [N] into K groups and [M] into K groups.

    Args:
        N, M: Number of row/column clusters.
        K: Number of fold groups per dimension.
        rng: NumPy random generator.

    Returns:
        Tuple of (row_groups, col_groups), each a list of index arrays.
    """
    row_perm = rng.permutation(N)
    col_perm = rng.permutation(M)

    row_groups: List[list] = [[] for _ in range(K)]
    col_groups: List[list] = [[] for _ in range(K)]

    for idx, r in enumerate(row_perm):
        row_groups[idx % K].append(r)
    for idx, c in enumerate(col_perm):
        col_groups[idx % K].append(c)

    return [np.array(g) for g in row_groups], [np.array(g) for g in col_groups]


def _variance_iid(scores: np.ndarray, n_obs: int) -> float:
    """Simple IID variance: Var(psi) / n."""
    return float(np.var(scores, ddof=1) / n_obs)


def _variance_twoway_cr(
    scores: np.ndarray, row_ids: np.ndarray, col_ids: np.ndarray,
    N: int, M: int
) -> float:
    """Two-way cluster robust variance — Cameron, Gelbach & Miller (2011).

    V(theta_hat) = V_row + V_col - V_iid
    """
    n_obs = len(scores)
    C = min(N, M)

    # V_row: cluster by row index
    row_sums = np.zeros(N)
    for i in range(N):
        mask = (row_ids == i)
        row_sums[i] = scores[mask].sum()

    dfc_row = N / (N - 1) if N > 1 else 1.0
    V_row = dfc_row * np.sum(row_sums ** 2) / n_obs ** 2

    # V_col: cluster by column index
    col_sums = np.zeros(M)
    for j in range(M):
        mask = (col_ids == j)
        col_sums[j] = scores[mask].sum()

    dfc_col = M / (M - 1) if M > 1 else 1.0
    V_col = dfc_col * np.sum(col_sums ** 2) / n_obs ** 2

    # V_iid: HC1 (diagonal terms counted in both row and col)
    dfc_iid = n_obs / (n_obs - 1) if n_obs > 1 else 1.0
    V_iid = dfc_iid * np.sum(scores ** 2) / n_obs ** 2

    # Cameron et al. (2011): V = V_row + V_col - V_iid
    V_twoway = V_row + V_col - V_iid

    return float(max(V_twoway, 1e-12))


def dml2_plm_iv(
    data: Dict[str, Any], strategy: str, ml_method: str = "lasso",
    K: int = 2, seed: int = None, variance_type: str = "twoway_cr",
    tau_true: float = 1.0,
) -> Dict[str, Any]:
    """DML2 estimator for the Partially Linear IV model.

    Implements Algorithm 1 from Chiang et al. (2021, arXiv:1909.03489v3):
    cross-fit nuisance estimates, solve orthogonal moment, compute SEs.

    Args:
        data: Dict with keys Y, D, X, Z, row_id, col_id.
        strategy: Cross-fitting strategy: 'no_cf', 'as_iid', 'chiang_k2'.
        ml_method: ML method for nuisance estimation.
        K: Number of fold groups per dimension.
        seed: Random seed.
        variance_type: 'iid' or 'twoway_cr'.

    Returns:
        Dict with tau_hat, se, covers, bias.
    """
    Y = data["Y"]
    D = data["D"]
    X = data["X"]
    Z = data["Z"]
    row_ids = data["row_id"]
    col_ids = data["col_id"]
    n_obs = len(Y)

    N = len(np.unique(row_ids))
    M = len(np.unique(col_ids))
    C = min(N, M)
    dim_x = X.shape[1]

    rng = np.random.default_rng(seed)
    make_ml = ML_METHODS[ml_method]

    g1_hat = np.zeros(n_obs)
    g2_hat = np.zeros(n_obs)
    m_hat = np.zeros(n_obs)

    if strategy == "no_cf":
        ml_g1 = make_ml(seed, dim_x).fit(X, Y)
        ml_g2 = make_ml(seed, dim_x).fit(X, D)
        ml_m = make_ml(seed, dim_x).fit(X, Z)

        g1_hat = ml_g1.predict(X)
        g2_hat = ml_g2.predict(X)
        m_hat = ml_m.predict(X)

    elif strategy == "as_iid":
        # Standard K-fold — Balkus "as-IID"
        indices = rng.permutation(n_obs)
        folds = np.array_split(indices, K)

        for k in range(K):
            te_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])

            ml_g1 = make_ml(seed, dim_x).fit(X[tr_idx], Y[tr_idx])
            ml_g2 = make_ml(seed, dim_x).fit(X[tr_idx], D[tr_idx])
            ml_m = make_ml(seed, dim_x).fit(X[tr_idx], Z[tr_idx])

            g1_hat[te_idx] = ml_g1.predict(X[te_idx])
            g2_hat[te_idx] = ml_g2.predict(X[te_idx])
            m_hat[te_idx] = ml_m.predict(X[te_idx])

    elif strategy == "chiang_k2":
        # K²-fold multiway cross-fitting — Chiang et al. (2021, Algorithm 1)
        row_groups, col_groups = _make_k2_partitions(N, M, K, rng)

        row_to_group = np.zeros(N, dtype=int)
        col_to_group = np.zeros(M, dtype=int)
        for k, grp in enumerate(row_groups):
            for r in grp:
                row_to_group[r] = k
        for l, grp in enumerate(col_groups):
            for c in grp:
                col_to_group[c] = l

        unit_row_group = row_to_group[row_ids]
        unit_col_group = col_to_group[col_ids]

        for k in range(K):
            for l in range(K):
                te_mask = (unit_row_group == k) & (unit_col_group == l)
                te_idx = np.where(te_mask)[0]

                if len(te_idx) == 0:
                    continue

                # Train: rows NOT in group k AND cols NOT in group l
                tr_mask = (unit_row_group != k) & (unit_col_group != l)
                tr_idx = np.where(tr_mask)[0]

                if len(tr_idx) < dim_x:
                    continue

                ml_g1 = make_ml(seed, dim_x).fit(X[tr_idx], Y[tr_idx])
                ml_g2 = make_ml(seed, dim_x).fit(X[tr_idx], D[tr_idx])
                ml_m = make_ml(seed, dim_x).fit(X[tr_idx], Z[tr_idx])

                g1_hat[te_idx] = ml_g1.predict(X[te_idx])
                g2_hat[te_idx] = ml_g2.predict(X[te_idx])
                m_hat[te_idx] = ml_m.predict(X[te_idx])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Solve DML2 moment condition — Chiang et al. (2021, eq. 2.5)
    Y_tilde = Y - g1_hat
    D_tilde = D - g2_hat
    Z_tilde = Z - m_hat

    numerator = np.sum(Y_tilde * Z_tilde)
    denominator = np.sum(D_tilde * Z_tilde)

    if abs(denominator) < 1e-10:
        theta_hat = np.nan
        se = np.nan
        covers = 0
        bias = np.nan
    else:
        theta_hat = numerator / denominator

        # Influence function — Chiang et al. (2021, eq. 2.5-2.6)
        eps_hat = Y_tilde - theta_hat * D_tilde
        psi = eps_hat * Z_tilde
        J_hat = -denominator / n_obs
        influence = -psi / J_hat

        # Always compute both SE types for decoupling analysis
        var_iid = _variance_iid(influence, n_obs)
        var_cr = _variance_twoway_cr(influence, row_ids, col_ids, N, M)

        se_iid = np.sqrt(var_iid)
        se_cr = np.sqrt(var_cr)

        # Primary SE uses the requested variance type
        se = se_cr if variance_type == "twoway_cr" else se_iid

        bias = theta_hat - tau_true
        covers = int(abs(bias) <= 1.96 * se)
        covers_iid = int(abs(bias) <= 1.96 * se_iid)
        covers_cr = int(abs(bias) <= 1.96 * se_cr)

    return {
        "tau_hat": theta_hat,
        "se": se,
        "se_iid": se_iid if 'se_iid' in dir() else np.nan,
        "se_cr": se_cr if 'se_cr' in dir() else np.nan,
        "covers": covers,
        "covers_iid": covers_iid if 'covers_iid' in dir() else 0,
        "covers_cr": covers_cr if 'covers_cr' in dir() else 0,
        "bias": theta_hat - tau_true,
    }
