"""Hurst exponent estimation for market structure analysis."""

from __future__ import annotations

import numpy as np


def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent using a log-log slope method.

    Parameters
    ----------
    series : np.ndarray
        One-dimensional time series.
    max_lag : int
        Maximum lag used in the estimation.

    Returns
    -------
    float
        Estimated Hurst exponent.

    Notes
    -----
    Interpretation:
        H ≈ 0.5  -> random walk
        H > 0.5  -> trending / persistent
        H < 0.5  -> mean reverting / anti-persistent
    """
    series = np.asarray(series, dtype=float)

    if series.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if len(series) < max_lag + 2:
        raise ValueError("series is too short for the chosen max_lag.")
    if max_lag < 3:
        raise ValueError("max_lag must be at least 3.")
    if not np.all(np.isfinite(series)):
        raise ValueError("series must contain only finite values.")

    lags = np.arange(2, max_lag + 1)
    tau = np.empty(len(lags), dtype=float)

    for i, lag in enumerate(lags):
        diff = series[lag:] - series[:-lag]
        tau[i] = np.std(diff, ddof=1)

    if np.any(tau <= 0.0):
        raise ValueError("non-positive tau encountered during Hurst estimation.")

    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)

    return float(slope)


def classify_hurst(H: float) -> str:
    """
    Classify market structure from Hurst exponent.
    """
    if not np.isfinite(H):
        raise ValueError("H must be finite.")

    if H > 0.55:
        return "TRENDING"
    if H < 0.45:
        return "MEAN_REVERTING"
    return "RANDOM_WALK"