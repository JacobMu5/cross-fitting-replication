"""Shared utility functions for the simulation framework."""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
