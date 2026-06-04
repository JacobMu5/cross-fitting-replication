"""R earth() wrapper via rpy2 — drop-in replacement for mars.Earth.

Calls the actual R earth package (compiled C/Fortran) through rpy2,
matching Balkus hard_cluster.R exactly:
  earth(A ~ X1..X5, degree=2, glm=list(family="binomial"))
  earth(Y ~ A + X1..X5, degree=2, glm=list(family="gaussian"))

Requires:
  - R installed and on PATH (https://cran.r-project.org/)
  - pip install rpy2
  - R earth package: Rscript -e "install.packages('earth')"

Reference R implementation:
  https://github.com/salbalkus/cross-fitting-dependent-data/blob/main/R/hard_cluster.R
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional

# Auto-detect R_HOME from the latest installed version
if "R_HOME" not in os.environ:
    _r_dirs = sorted(Path(r"C:\Program Files\R").glob("R-*"), reverse=True)
    if _r_dirs:
        os.environ["R_HOME"] = str(_r_dirs[0])

os.environ.setdefault("LC_ALL", "C")  # suppress locale encoding errors on Windows

import rpy2.robjects as ro

ro.r('suppressPackageStartupMessages(library(earth))')

# Counter for unique R variable names (avoids collisions in parallel)
_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f".e{_counter}"


class Earth:
    """R earth() wrapper with sklearn-like interface.

    Drop-in replacement for mars.Earth. Uses R's compiled C/Fortran
    earth implementation for ~40x speedup over pure-Python mars.py.

    Args:
        max_degree: Maximum interaction degree (R: degree).
        max_terms: Maximum basis functions (R: nk). None = R default.
        penalty: GCV penalty per knot (R default: 3 for degree>1).
        glm_family: None/'gaussian' or 'binomial'.
    """

    def __init__(self, max_degree: int = 2, max_terms: Optional[int] = None,
                 penalty: Optional[float] = None,
                 glm_family: Optional[str] = None):
        self.max_degree = max_degree
        self.max_terms = max_terms
        self.penalty = penalty
        self.glm_family = glm_family
        self._r_model_name: Optional[str] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Earth":
        """Fit MARS model via R earth().

        Args:
            X: Feature matrix (n, p).
            y: Response vector (n,).

        Returns:
            self
        """
        if self.penalty is None:
            self.penalty = 3.0 if self.max_degree > 1 else 2.0

        n, p = X.shape
        xid, yid = _next_id(), _next_id()

        # Push numpy data into R global environment
        ro.globalenv[xid] = ro.r.matrix(
            ro.FloatVector(X.flatten(order="F")), nrow=n, ncol=p
        )
        ro.globalenv[yid] = ro.FloatVector(y.ravel())

        # Build R expression with literal values — avoids all quoting issues
        parts = [
            f"earth(x={xid}, y={yid}",
            f"degree={self.max_degree}L",
            f"penalty={self.penalty}",
        ]

        if self.max_terms is not None:
            parts.append(f"nk={self.max_terms}L")

        if self.glm_family is not None:
            parts.append(f"glm=list(family='{self.glm_family}')")

        r_code = ", ".join(parts) + ")"

        # Fit and store model in R globalenv (keeps it as opaque R object)
        self._r_model_name = _next_id()
        ro.globalenv[self._r_model_name] = ro.r(r_code)

        # Clean up input data from R env
        ro.r(f"rm({xid}, {yid})")
        return self

    def _predict_r(self, X: np.ndarray, r_type: str = "link") -> np.ndarray:
        """Push X to R, call predict(), pull result, clean up."""
        n, p = X.shape
        xid = _next_id()
        ro.globalenv[xid] = ro.r.matrix(
            ro.FloatVector(X.flatten(order="F")), nrow=n, ncol=p
        )
        preds = np.array(list(
            ro.r(f"as.numeric(predict({self._r_model_name}, {xid}, type='{r_type}'))")
        ))
        ro.r(f"rm({xid})")
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict response for new data X."""
        return self._predict_r(X, r_type="link")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(y=1|X) for binomial GLM. Returns (n, 2) sklearn-style."""
        if self.glm_family != "binomial":
            raise ValueError("predict_proba requires glm_family='binomial'")
        p1 = self._predict_r(X, r_type="response")
        return np.column_stack([1 - p1, p1])
