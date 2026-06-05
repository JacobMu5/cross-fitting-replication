"""Chiang replication scenarios — builders for PLM-IV DML2 experiments."""

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scenarios import ScenarioConfig
from src.dgps.chiang_exact import generate_chiang_exact
from src.estimators.plm_iv_dml import dml2_plm_iv


def _chiang_estimator_adapter(
    df: dict[str, Any], strategy: str, ml_method: str, K: int,
    variance_type: str, seed: int, tau_true: float
) -> dict[str, Any]:
    """Adapter bridging the generic runner interface to dml2_plm_iv."""
    return dml2_plm_iv(
        data=df,
        strategy=strategy,
        ml_method=ml_method,
        K=K,
        seed=seed,
        variance_type=variance_type,
        tau_true=tau_true,
    )


# Chiang (2021) scenario definitions

CHIANG_SCENARIO_DEFS = [
    # Lasso (Chiang's original learner — Donsker-class)
    {"N": 25, "M": 25, "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(25,25), dim=100, K^2=4, Lasso"},
    {"N": 25, "M": 25, "dim_x": 200, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(25,25), dim=200, K^2=4, Lasso"},
    {"N": 50, "M": 50, "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(50,50), dim=100, K^2=4, Lasso"},
    {"N": 50, "M": 50, "dim_x": 200, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(50,50), dim=200, K^2=4, Lasso"},
    {"N": 25, "M": 25, "dim_x": 100, "K": 3, "ml": "lasso", "s_eu": 0.25,
     "label": "(25,25), dim=100, K^2=9, Lasso"},
    {"N": 25, "M": 25, "dim_x": 200, "K": 3, "ml": "lasso", "s_eu": 0.25,
     "label": "(25,25), dim=200, K^2=9, Lasso"},
    # Random Forest (non-Donsker — tests whether CF matters on Chiang's DGP)
    {"N": 25, "M": 25, "dim_x": 100, "K": 2, "ml": "rf", "s_eu": 0.25,
     "label": "(25,25), dim=100, K^2=4, RF"},
]

CHIANG_STRATEGIES = ["no_cf", "as_iid", "chiang_k2"]


def get_chiang_scenarios(
    n_sim: int = 500,
    first_seed: int = 7691,
    strategy_filter: str | None = None,
) -> list[ScenarioConfig]:
    """Build ScenarioConfig list for Chiang et al. (2021) replication."""
    scenarios: list[ScenarioConfig] = []

    for sc in CHIANG_SCENARIO_DEFS:
        for strategy in CHIANG_STRATEGIES:
            if strategy_filter and strategy != strategy_filter:
                continue

            # For As-IID, use K² folds to match total fold count
            K_use = sc["K"] ** 2 if strategy == "as_iid" else sc["K"]

            name = (f"chiang__{sc['N']}x{sc['M']}__dim{sc['dim_x']}"
                    f"__K{sc['K']}__{strategy}")

            scenarios.append(ScenarioConfig(
                name=name,
                dgp_func=generate_chiang_exact,
                dgp_kwargs=dict(
                    N=sc["N"], M=sc["M"],
                    dim_x=sc["dim_x"], s_eu=sc["s_eu"],
                ),
                estimator_func=_chiang_estimator_adapter,
                estimator_kwargs=dict(
                    strategy=strategy,
                    ml_method=sc["ml"],
                    K=K_use,
                    variance_type="twoway_cr",
                ),
                n_simulations=n_sim,
                first_seed=first_seed,
                metadata=dict(
                    scenario=sc["label"],
                    N=sc["N"], M=sc["M"],
                    K=sc["K"],
                    ml_method=sc["ml"],
                    s_eu=sc["s_eu"],
                    strategy=strategy,
                    var_type="twoway_cr",
                ),
            ))

    return scenarios


# Chen & Chiang (2026) convergence scenario definitions

CHEN_CHIANG_SCENARIO_DEFS = [
    {"N": 25,  "M": 25,  "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(25,25), n=625"},
    {"N": 50,  "M": 50,  "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(50,50), n=2500"},
    {"N": 75,  "M": 75,  "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(75,75), n=5625"},
    {"N": 100, "M": 100, "dim_x": 100, "K": 2, "ml": "lasso", "s_eu": 0.25,
     "label": "(100,100), n=10000"},
]


def get_chen_chiang_scenarios(
    n_sim: int = 200,
    first_seed: int = 7691,
    strategy_filter: str | None = None,
) -> list[ScenarioConfig]:
    """Build ScenarioConfig list for Chen & Chiang (2026) convergence test."""
    scenarios: list[ScenarioConfig] = []

    for sc in CHEN_CHIANG_SCENARIO_DEFS:
        for strategy in CHIANG_STRATEGIES:
            if strategy_filter and strategy != strategy_filter:
                continue

            K_use = sc["K"] ** 2 if strategy == "as_iid" else sc["K"]

            name = (f"chen_chiang__{sc['N']}x{sc['M']}"
                    f"__dim{sc['dim_x']}__{strategy}")

            scenarios.append(ScenarioConfig(
                name=name,
                dgp_func=generate_chiang_exact,
                dgp_kwargs=dict(
                    N=sc["N"], M=sc["M"],
                    dim_x=sc["dim_x"], s_eu=sc["s_eu"],
                ),
                estimator_func=_chiang_estimator_adapter,
                estimator_kwargs=dict(
                    strategy=strategy,
                    ml_method=sc["ml"],
                    K=K_use,
                    variance_type="twoway_cr",
                ),
                n_simulations=n_sim,
                first_seed=first_seed,
                metadata=dict(
                    scenario=sc["label"],
                    N=sc["N"], M=sc["M"],
                    n=sc["N"] * sc["M"],
                    K=sc["K"],
                    ml_method=sc["ml"],
                    s_eu=sc["s_eu"],
                    strategy=strategy,
                    var_type="twoway_cr",
                ),
            ))

    return scenarios
