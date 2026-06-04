"""Type protocols for DGPs and estimators (PEP 544 structural subtyping).

The runner calls DGPs and estimators as plain functions:
    df, tau_true = dgp_func(**dgp_kwargs, seed=seed)
    result_dict  = estimator_func(df=df, **estimator_kwargs, seed=seed, tau_true=tau_true)

Any callable matching these signatures satisfies the protocol.
"""

from typing import Any, Protocol


class DGPProtocol(Protocol):
    """Protocol for Data Generating Processes.

    A DGP is any callable that accepts keyword arguments (forwarded from
    ScenarioConfig.dgp_kwargs) plus a seed, and returns (data, true_effect).
    """

    def __call__(self, *, seed: int, **kwargs: Any) -> tuple[Any, float]: ...


class EstimatorProtocol(Protocol):
    """Protocol for causal estimators.

    An estimator is any callable that accepts data (from a DGP), keyword
    arguments (forwarded from ScenarioConfig.estimator_kwargs), a seed,
    and the true treatment effect, and returns a result dict.
    """

    def __call__(
        self, *, df: Any, seed: int, tau_true: float, **kwargs: Any
    ) -> dict[str, Any]: ...
