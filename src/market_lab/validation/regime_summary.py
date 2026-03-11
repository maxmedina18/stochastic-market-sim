"""Helpers for summarizing current volatility regime."""

from __future__ import annotations

import numpy as np


def current_regime_label(regimes: np.ndarray) -> str:
    """
    Return the latest volatility regime label.
    """
    regimes = np.asarray(regimes)

    if regimes.ndim != 1:
        raise ValueError("regimes must be one-dimensional.")
    if len(regimes) == 0:
        raise ValueError("regimes must not be empty.")

    return "HIGH_VOL" if int(regimes[-1]) == 1 else "LOW_VOL"