"""Volatility regime detection."""

from __future__ import annotations

import numpy as np


def detect_volatility_regimes(
    rolling_vol: np.ndarray,
    threshold_quantile: float = 0.75,
) -> np.ndarray:
    """
    Classify volatility regimes based on a quantile threshold.

    Parameters
    ----------
    rolling_vol : np.ndarray
        Rolling volatility series.
    threshold_quantile : float
        Quantile used to separate regimes.

    Returns
    -------
    np.ndarray
        Array of regime labels:
        0 = low volatility
        1 = high volatility
    """

    rolling_vol = np.asarray(rolling_vol, dtype=float)

    if rolling_vol.ndim != 1:
        raise ValueError("rolling_vol must be one-dimensional.")

    threshold = np.quantile(rolling_vol, threshold_quantile)

    regimes = (rolling_vol > threshold).astype(int)

    return regimes