"""Empirical stylized facts of financial markets."""

from __future__ import annotations

import numpy as np


def rolling_volatility(
    returns: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Compute rolling volatility of returns.

    Parameters
    ----------
    returns : np.ndarray
        One-dimensional array of returns.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling volatility series.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional.")
    if window <= 1:
        raise ValueError("window must be > 1.")
    if len(returns) < window:
        raise ValueError("window is too large for the return series.")

    vol = []

    for i in range(window, len(returns) + 1):
        segment = returns[i - window : i]
        vol.append(np.std(segment, ddof=1))

    return np.array(vol)


def autocorrelation(
    series: np.ndarray,
    lag: int,
) -> float:
    """
    Compute autocorrelation at a given lag.
    """
    series = np.asarray(series, dtype=float)

    if series.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if lag <= 0:
        raise ValueError("lag must be positive.")
    if lag >= len(series):
        raise ValueError("lag too large.")

    x = series[:-lag]
    y = series[lag:]

    return float(np.corrcoef(x, y)[0, 1])


def squared_return_autocorrelation(
    returns: np.ndarray,
    lag: int,
) -> float:
    """
    Compute autocorrelation of squared returns.
    Used to detect volatility clustering.
    """
    returns = np.asarray(returns, dtype=float)
    return autocorrelation(returns**2, lag)