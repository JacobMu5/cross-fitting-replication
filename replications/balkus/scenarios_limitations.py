"""Limitations experiment: dense full-grid DGP (Corollary 2 violated).

Separate from the main bounded-dependence factorial. Tests what happens
when the bounded-cluster assumption fails: full N×M grid where row
degree = M and column degree = N grow with the sample.

Grid sizes: 20×20 (n=400), 40×40 (n=1600), 60×60 (n=3600).
  - 20×20 matches the smallest grid in the main factorial for comparison.
  - 40×40, 60×60 show behavior as the degree diverges.

Only deep_rf learner — the leakage mechanism requires a data-adaptive
learner that can exploit shared cluster membership to denoise proxies.
"""

from src.scenarios import ScenarioConfig
from src.dgps.balkus_fullgrid import generate_balkus_fullgrid
from src.estimators.dml import crossfit_dml


_FULLGRID_SIZES = [
    (20, 20),    # n=400,  row degree=20  — comparable to main factorial
    (40, 40),    # n=1600, row degree=40  — moderate violation
    (60, 60),    # n=3600, row degree=60  — strong violation
]

_STRATEGIES = ["no_cf", "as_iid", "chiang"]


def get_limitations_scenarios(
    n_sim: int = 200,
    first_seed: int = 7691,
) -> list[ScenarioConfig]:
    """Build the fullgrid limitations factorial.

    Args:
        n_sim: Monte Carlo replications per scenario.
        first_seed: Starting random seed.

    Returns:
        List of ScenarioConfig for the fullgrid experiment.
    """
    scenarios: list[ScenarioConfig] = []

    for N, M in _FULLGRID_SIZES:
        for strategy in _STRATEGIES:
            name = f"balkus_fullgrid__deep_rf__N{N}_M{M}__{strategy}"
            scenarios.append(ScenarioConfig(
                name=name,
                dgp_func=generate_balkus_fullgrid,
                dgp_kwargs=dict(N=N, M=M),
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
                    dgp="balkus_fullgrid",
                    dgp_label="Dense Full-Grid",
                    learner="deep_rf",
                    learner_label="Deep RF",
                    strategy=strategy,
                    N=N,
                    M=M,
                ),
            ))

    return scenarios
