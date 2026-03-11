"""GARCH(1,1) volatility model."""

from __future__ import annotations

import numpy as np


def validate_garch_parameters(
    omega: float,
    alpha: float,
    beta: float,
) -> None:
    """
    Validate GARCH(1,1) parameters.
    """
    params = [omega, alpha, beta]

    if not all(np.isfinite(p) for p in params):
        raise ValueError("GARCH parameters must be finite.")

    if omega <= 0.0:
        raise ValueError("omega must be positive.")

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be less than 1 for covariance stationarity.")


def garch11_variance(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute conditional variance under a GARCH(1,1) model.

    Parameters
    ----------
    returns : np.ndarray
        One-dimensional return series.
    omega : float
        Constant term.
    alpha : float
        Shock sensitivity.
    beta : float
        Volatility persistence.

    Returns
    -------
    np.ndarray
        Conditional variance series.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional.")

    if returns.size < 2:
        raise ValueError("returns must contain at least two observations.")

    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values.")

    validate_garch_parameters(omega=omega, alpha=alpha, beta=beta)

    variances = np.zeros_like(returns)

    unconditional_variance = omega / (1.0 - alpha - beta)
    variances[0] = unconditional_variance

    for t in range(1, len(returns)):
        variances[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + beta * variances[t - 1]
        )

    return variances


def garch11_volatility(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute conditional volatility under a GARCH(1,1) model.
    """
    variances = garch11_variance(
        returns=returns,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )
    return np.sqrt(variances)