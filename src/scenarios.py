"""ScenarioConfig dataclass — single source of truth for simulation parameters."""

from dataclasses import dataclass, field
from typing import Any

from src.protocols import DGPProtocol, EstimatorProtocol


@dataclass
class ScenarioConfig:
    """Fully specifies a single simulation scenario.

    Attributes:
        name: Unique identifier (used for resume and results grouping).
        dgp_func: DGP generator with signature (..., seed=int) -> (df, tau).
        dgp_kwargs: All kwargs for dgp_func EXCEPT seed (supplied by runner).
        estimator_func: Estimator with signature (df, ..., seed, tau_true) -> dict.
        estimator_kwargs: All kwargs for estimator_func EXCEPT df, seed, tau_true.
        n_simulations: Number of Monte Carlo replications.
        first_seed: Starting seed; rep i uses seed = first_seed + i.
        metadata: Extra columns to attach to every result row.
        pate: Population ATE (precomputed from a large grid). If set, the runner
              evaluates bias/coverage against this instead of per-sample SATE.
    """
    name: str
    dgp_func: DGPProtocol
    dgp_kwargs: dict[str, Any]
    estimator_func: EstimatorProtocol
    estimator_kwargs: dict[str, Any]
    n_simulations: int
    first_seed: int = 7691
    metadata: dict[str, Any] = field(default_factory=dict)
    pate: float | None = None
