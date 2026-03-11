"""Utilities for converting price series into return series."""

from __future__ import annotations

import numpy as np


def validate_price_series(prices: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a one-dimensional price series.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices.

    Returns
    -------
    np.ndarray
        Validated price array as float.

    Raises
    ------
    ValueError
        If prices is not one-dimensional, has fewer than two observations,
        contains non-finite values, or contains non-positive prices.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be a one-dimensional array.")
    if prices.size < 2:
        raise ValueError("prices must contain at least two observations.")
    if not np.all(np.isfinite(prices)):
        raise ValueError("prices must contain only finite values.")
    if not np.all(prices > 0.0):
        raise ValueError("prices must contain only positive values.")

    return prices


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute simple returns from a price series.

    Parameters
    ----------
    prices : np.ndarray
        One-dimensional array of prices.

    Returns
    -------
    np.ndarray
        Simple returns r_t = P_t / P_{t-1} - 1.
    """
    prices = validate_price_series(prices)
    return prices[1:] / prices[:-1] - 1.0


def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from a price series.

    Parameters
    ----------
    prices : np.ndarray
        One-dimensional array of prices.

    Returns
    -------
    np.ndarray
        Log returns log(P_t / P_{t-1}).
    """
    prices = validate_price_series(prices)
    return np.log(prices[1:] / prices[:-1])


def annualized_volatility_from_log_returns(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized volatility from log returns.

    Parameters
    ----------
    returns : np.ndarray
        One-dimensional array of log returns.
    periods_per_year : int
        Number of return periods in one year.

    Returns
    -------
    float
        Annualized volatility.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be a one-dimensional array.")
    if returns.size < 2:
        raise ValueError("returns must contain at least two observations.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))