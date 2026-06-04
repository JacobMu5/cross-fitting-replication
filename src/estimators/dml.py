"""DML estimator faithful to Balkus et al. (2026) hard_cluster.R.

Balkus calls this "IID DML" (as_iid) and "2-Way DML" (chiang).
Uses the IRM (Interactive Regression Model) variant of DML — not the PLR
"double residual" approach — because the treatment effect is heterogeneous.
The score is the standard doubly-robust / AIPW influence function.

Architecture matching hard_cluster.R:
  - Joint outcome model: earth(Y ~ A + X1..X5), counterfactuals via A=1/A=0
  - K=2 folds and propensity clip=1e-3

Reference: https://github.com/salbalkus/cross-fitting-dependent-data/blob/main/R/hard_cluster.R
"""

import numpy as np
from typing import Any, Dict

from sklearn.model_selection import KFold


# Learner factory

def _featurize(X: np.ndarray) -> np.ndarray:
    """Replicate new_cluster.R featurize(): X² + sin(X) + pairwise interactions.

    Input: (n, p) with p >= 5 columns.
    Output: (n, p + 10) — original + 3 squares + 3 sines + 4 interactions.
    """
    out = [X,
           X[:, 0:3] ** 2,                                       # X1², X2², X3²
           np.sin(X[:, 0:3]),                                     # sin(X1..X3)
           (X[:, 0] * X[:, 1])[:, None],                          # X1·X2
           (X[:, 1] * X[:, 2])[:, None],                          # X2·X3
           (X[:, 2] * X[:, 3])[:, None],                          # X3·X4
           (X[:, 3] * X[:, 4])[:, None]]                          # X4·X5
    return np.hstack(out)


def _make_learner(learner_type: str, task: str, seed: int):
    """Create a learner instance.

    Args:
        learner_type: 'earth_exact', 'deep_rf', or 'lasso'.
        task: 'reg' (outcome regression) or 'cls' (propensity classification).
        seed: Random seed (used by RF/Lasso, ignored by Earth).
    """
    if learner_type == "earth_exact":
        from src.estimators.mars_r import Earth
        family = "gaussian" if task == "reg" else "binomial"
        return Earth(max_degree=2, penalty=3.0, glm_family=family)

    elif learner_type == "deep_rf":
        if task == "reg":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                random_state=seed, n_jobs=1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                random_state=seed, n_jobs=1,
            )

    elif learner_type == "lasso":
        # Matching new_cluster.R: cv.glmnet with featurize() preprocessing
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        if task == "reg":
            from sklearn.linear_model import LassoCV
            return Pipeline([
                ("featurize", FunctionTransformer(_featurize)),
                ("scale", StandardScaler()),
                ("lasso", LassoCV(cv=5, random_state=seed)),
            ])
        else:
            from sklearn.linear_model import LogisticRegressionCV
            return Pipeline([
                ("featurize", FunctionTransformer(_featurize)),
                ("scale", StandardScaler()),
                ("logistic", LogisticRegressionCV(
                    cv=5, penalty="l1", solver="saga",
                    max_iter=2000, random_state=seed,
                )),
            ])

    raise ValueError(f"Unknown learner: {learner_type}")


# DR score — Balkus R: dr <- A*(Y-Q1)/g - (1-A)*(Y-Q0)/(1-g) + (Q1-Q0)

def _dr_scores(
    Y: np.ndarray, A: np.ndarray,
    mu1: np.ndarray, mu0: np.ndarray,
    e: np.ndarray, clip: float = 1e-3,
) -> np.ndarray:
    """Doubly-robust influence-function scores. Clip matches Balkus R (1e-3)."""
    e = np.clip(e, clip, 1 - clip)
    return mu1 - mu0 + A * (Y - mu1) / e - (1 - A) * (Y - mu0) / (1 - e)


# Nuisance fitting (joint outcome model)

def _fit_nuisance(
    X_tr: np.ndarray, A_tr: np.ndarray, Y_tr: np.ndarray,
    X_te: np.ndarray, learner_type: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit both nuisance models on train fold, predict on test fold.

    "Double" = two ML models:
      1. Propensity ê(X) = P(A=1|X) — earth(A ~ X, glm=binomial)
      2. Outcome μ̂(A,X) = E[Y|A,X] — earth(Y ~ A+X, glm=gaussian)
         Counterfactuals via A-intervention: predict with A=1 and A=0

    Returns:
        (mu1_hat, mu0_hat, e_hat) arrays over the test fold.
    """
    n_te = len(X_te)

    cls = _make_learner(learner_type, "cls", seed)
    cls.fit(X_tr, A_tr)
    e_hat = cls.predict_proba(X_te)[:, 1]

    XA_tr = np.column_stack([A_tr, X_tr])
    reg = _make_learner(learner_type, "reg", seed)
    reg.fit(XA_tr, Y_tr)

    mu1_hat = reg.predict(np.column_stack([np.ones(n_te), X_te]))
    mu0_hat = reg.predict(np.column_stack([np.zeros(n_te), X_te]))

    return mu1_hat, mu0_hat, e_hat


# Fold generators

def _folds_no_cf(n: int, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    """No cross-fitting: train on all, predict on all."""
    idx = np.arange(n)
    return [(idx, idx)]


def _folds_as_iid(n: int, seed: int, K: int = 2, **_) -> list[tuple[np.ndarray, np.ndarray]]:
    """IID KFold — Balkus R: iid_cf()."""
    return list(KFold(n_splits=K, shuffle=True, random_state=seed).split(np.arange(n)))


def _folds_chiang(
    n: int, seed: int, row_ids: np.ndarray,
    col_ids: np.ndarray | None, K: int = 2, **_,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K²-fold two-way cross-fitting — Chiang et al. (2021, Algorithm 1)."""
    rng = np.random.default_rng(seed)
    unique_rows = np.unique(row_ids)
    unique_cols = np.unique(col_ids) if col_ids is not None else np.array([0])

    K_r, K_c = min(K, len(unique_rows)), min(K, len(unique_cols))

    # Assign clusters to folds — Balkus R: sample(rep(1:K, length.out = N))
    row_fold_map = {r: i % K_r for i, r in enumerate(rng.permutation(unique_rows))}
    col_fold_map = {c: i % K_c for i, c in enumerate(rng.permutation(unique_cols))}

    unit_row_fold = np.array([row_fold_map[r] for r in row_ids])
    unit_col_fold = (
        np.array([col_fold_map[c] for c in col_ids])
        if col_ids is not None
        else np.zeros(n, dtype=int)
    )

    folds = []
    for kr in range(K_r):
        for kc in range(K_c):
            te = np.where((unit_row_fold == kr) & (unit_col_fold == kc))[0]
            # Strict complement — Balkus R: train <- !(dt$rf == r | dt$cf == c)
            tr = np.where((unit_row_fold != kr) & (unit_col_fold != kc))[0]
            folds.append((tr, te))
    return folds


FOLD_STRATEGIES = {
    "no_cf": _folds_no_cf,
    "as_iid": _folds_as_iid,
    "chiang": _folds_chiang,
}


# Cluster-robust SE — Cameron, Gelbach & Miller (2011)

def _cluster_robust_se(
    psi: np.ndarray, row_ids: np.ndarray, col_ids: np.ndarray | None,
) -> float:
    """Two-way cluster-robust SE: V_cr = V_row + V_col - V_iid.

    Cameron, Gelbach & Miller (2011, eq. 2.2). Ensures SE accounts
    for within-cluster correlation of influence-function scores.

    Args:
        psi: Centered DR scores (n,).
        row_ids: Row cluster IDs (n,).
        col_ids: Column cluster IDs (n,), or None for one-way.

    Returns:
        Cluster-robust standard error.
    """
    n = len(psi)

    # Row-clustered variance
    v_row = sum(
        psi[row_ids == g].sum() ** 2 for g in np.unique(row_ids)
    ) / n**2

    # IID variance
    v_iid = np.sum(psi**2) / n**2

    if col_ids is not None:
        # Column-clustered variance
        v_col = sum(
            psi[col_ids == g].sum() ** 2 for g in np.unique(col_ids)
        ) / n**2
        v_cr = v_row + v_col - v_iid
    else:
        v_cr = v_row

    return float(np.sqrt(max(v_cr, 0.0)))


# Main estimator

def crossfit_dml(
    df: Any, strategy: str, learner_type: str, seed: int, tau_true: float,
    n_folds: int = 2,
) -> Dict[str, Any]:
    """DML-IRM estimator with joint outcome model, faithful to Balkus R code.

    Args:
        df: DataFrame with columns X1–X5 (or L1–L5), A, Y, row_id, col_id.
        strategy: 'no_cf' | 'as_iid' | 'chiang'.
        learner_type: 'earth_exact' | 'deep_rf' | 'lasso'.
        seed: Random seed.
        tau_true: True ATE for coverage calculation.
        n_folds: Number of cross-fitting folds (K=2 for MARS, K=5 for LASSO).

    Returns:
        Dict with tau_hat, se/se_iid, se_cr, covers/covers_iid, covers_cr, bias.
    """
    if "X1" in df.columns:
        X_cols = sorted([c for c in df.columns if c.startswith("X")],
                        key=lambda c: int(c[1:]))
    else:
        X_cols = sorted([c for c in df.columns if c.startswith("L")],
                        key=lambda c: int(c[1:]))

    X = df[X_cols].values
    Y = df["Y"].values
    A = df["A"].values
    row_ids = df["row_id"].values
    col_ids = df["col_id"].values if "col_id" in df.columns else None
    n = len(Y)

    if strategy not in FOLD_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    folds = FOLD_STRATEGIES[strategy](
        n=n, seed=seed, row_ids=row_ids, col_ids=col_ids, K=n_folds,
    )

    mu1h, mu0h, eh = np.zeros(n), np.zeros(n), np.full(n, 0.5)
    for tr, te in folds:
        if len(te) == 0 or len(tr) < 10:
            continue
        mu1h[te], mu0h[te], eh[te] = _fit_nuisance(
            X[tr], A[tr], Y[tr], X[te], learner_type, seed
        )

    scores = _dr_scores(Y, A, mu1h, mu0h, eh)
    tau_hat = float(scores.mean())

    # IID SE — Balkus R: se = sqrt(mean(psi^2)) / sqrt(n)
    psi = scores - tau_hat
    se_iid = float(np.sqrt(np.mean(psi ** 2)) / np.sqrt(n))
    covers_iid = int(abs(tau_hat - tau_true) <= 1.96 * se_iid)

    # Cluster-robust SE — Cameron, Gelbach & Miller (2011)
    se_cr = _cluster_robust_se(psi, row_ids, col_ids)
    covers_cr = int(abs(tau_hat - tau_true) <= 1.96 * se_cr)

    return {
        "tau_hat": tau_hat,
        "se": se_iid,          # backward compat
        "se_iid": se_iid,
        "se_cr": se_cr,
        "covers": covers_iid,  # backward compat
        "covers_iid": covers_iid,
        "covers_cr": covers_cr,
        "bias": tau_hat - tau_true,
    }

