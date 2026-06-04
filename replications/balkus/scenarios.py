"""Balkus replication scenarios -- ScenarioConfig builders.

Two scenario families:

1. Full-grid DGPs (original Balkus replication):
   balkus_original, balkus_nocompress, shared_effects.
   Grid sizes N=M in {20, 30, 50}. All learners. Analytical PATE.

2. Bounded-cluster adversarial DGPs (stress test):
   balkus_adversarial (moderate effects), balkus_extreme (amplified effects).
   Sparse d-regular bipartite graph, N in {100, 200, 500}, d=20.
   deep_rf only. Replicate-specific SATE (pate=None) because the
   0.4*U*V term in tau causes large finite-sample ATE fluctuation.
"""

from src.scenarios import ScenarioConfig
from src.dgps.balkus_original import generate_balkus_original
from src.dgps.balkus_nocompress import generate_balkus_nocompress
from src.dgps.shared_effects import generate_shared_effects
from src.dgps.balkus_adversarial import generate_balkus_adversarial
from src.dgps.balkus_extreme import generate_balkus_extreme
from src.estimators.dml import crossfit_dml


# ── Full-grid DGPs ──────────────────────────────────────────────────

_GRID_SIZES = [
    (20, 20),
    (30, 30),
    (50, 50),
]

# Analytical PATE = E[tau(X)] for each full-grid DGP
# Derivation: tau = 1 + 0.5sin(X1) - 0.5sin(X2) + 0.3·X3·X4 + 0.4·a_t·b_t
#   E[sin(X_k)] = 0 (odd function of symmetric distribution)
#   E[a_t·b_t] = 0 (independent mean-zero)
#   E[0.3·X3·X4] = 0.3·E[0.3·X2²·1(X1>0)] = 0.3·0.3·1·0.5 = 0.045
_PATE = {
    "balkus_original":   1.045,   # 1 + 0.045
    "balkus_nocompress": 0.845,   # 1.045 - 0.4·P(X2>0) = 1.045 - 0.2
    "shared_effects":    1.000,   # tau is constant = 1
}

_GRID_DGP_CONFIGS = [
    (generate_balkus_original,   "balkus_original",   "Balkus Original"),
    (generate_balkus_nocompress, "balkus_nocompress",  "No Compression"),
    (generate_shared_effects,    "shared_effects",    "Shared Effects"),
]

# (learner_type, label, n_folds)
_LEARNER_CONFIGS = [
    ("earth_exact", "MARS Baseline",  2),   # hard_cluster.R: K=2
    ("deep_rf",     "Deep RF",        2),
    ("lasso",       "LASSO",          5),   # new_cluster.R: K=5
]

_STRATEGIES = ["no_cf", "as_iid", "chiang"]


# ── Bounded-cluster adversarial DGPs ────────────────────────────────

_ADVERSARIAL_GRID = [
    (100, 20),   # n = 2,000
    (200, 20),   # n = 4,000
    (500, 20),   # n = 10,000
]

_ADVERSARIAL_DGP_CONFIGS = [
    (generate_balkus_adversarial, "balkus_adversarial", "Bounded Adversarial"),
    (generate_balkus_extreme,     "balkus_extreme",     "Bounded Extreme"),
]


def get_balkus_scenarios(
    n_sim: int = 200,
    first_seed: int = 7691,
    dgp_filter: str | None = None,
    learner_filter: str | None = None,
    strategy_filter: str | None = None,
) -> list[ScenarioConfig]:
    """Build the full factorial: learner x DGP x strategy x grid.

    Args:
        n_sim: Number of Monte Carlo replications per scenario.
        first_seed: Starting seed for reproducibility.
        dgp_filter: If set, only include this DGP name.
        learner_filter: If set, only include this learner.
        strategy_filter: If set, only include this strategy.

    Returns:
        List of ScenarioConfig objects.
    """
    scenarios: list[ScenarioConfig] = []

    # ── Full-grid DGPs (analytical PATE) ────────────────────────────
    for dgp_func, dgp_name, dgp_label in _GRID_DGP_CONFIGS:
        if dgp_filter and dgp_name != dgp_filter:
            continue

        for learner, learner_label, n_folds in _LEARNER_CONFIGS:
            if learner_filter and learner != learner_filter:
                continue

            for N, M in _GRID_SIZES:
                for strategy in _STRATEGIES:
                    if strategy_filter and strategy != strategy_filter:
                        continue

                    name = (f"{dgp_name}__{learner}__N{N}__{strategy}"
                            .replace(" ", "_"))
                    scenarios.append(ScenarioConfig(
                        name=name,
                        dgp_func=dgp_func,
                        dgp_kwargs=dict(N=N, M=M),
                        estimator_func=crossfit_dml,
                        estimator_kwargs=dict(
                            strategy=strategy,
                            learner_type=learner,
                            n_folds=n_folds,
                        ),
                        n_simulations=n_sim,
                        first_seed=first_seed,
                        pate=_PATE[dgp_name],
                        metadata=dict(
                            dgp=dgp_name,
                            dgp_label=dgp_label,
                            learner=learner,
                            learner_label=learner_label,
                            strategy=strategy,
                            N=N,
                            M=M,
                        ),
                    ))

    # ── Bounded-cluster adversarial DGPs (replicate-specific SATE) ──
    for dgp_func, dgp_name, dgp_label in _ADVERSARIAL_DGP_CONFIGS:
        if dgp_filter and dgp_name != dgp_filter:
            continue
        # deep_rf only for adversarial — MARS/lasso don't add signal
        if learner_filter and learner_filter != "deep_rf":
            continue

        for N, d in _ADVERSARIAL_GRID:
            for strategy in _STRATEGIES:
                if strategy_filter and strategy != strategy_filter:
                    continue

                name = (f"{dgp_name}__deep_rf__N{N}_d{d}__{strategy}"
                        .replace(" ", "_"))
                scenarios.append(ScenarioConfig(
                    name=name,
                    dgp_func=dgp_func,
                    dgp_kwargs=dict(N=N, d=d),
                    estimator_func=crossfit_dml,
                    estimator_kwargs=dict(
                        strategy=strategy,
                        learner_type="deep_rf",
                        n_folds=2,
                    ),
                    n_simulations=n_sim,
                    first_seed=first_seed,
                    pate=None,  # replicate-specific SATE
                    metadata=dict(
                        dgp=dgp_name,
                        dgp_label=dgp_label,
                        learner="deep_rf",
                        learner_label="Deep RF",
                        strategy=strategy,
                        N=N,
                        M=d,  # M=d for consistent column naming
                    ),
                ))

    return scenarios
