"""Distribution diagnostics for financial returns."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def distribution_summary(returns: np.ndarray) -> dict[str, float]:
    """
    Compute basic distribution diagnostics for a return series.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional.")
    if len(returns) < 2:
        raise ValueError("returns must contain at least two observations.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values.")

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    skewness = float(skew(returns))
    excess_kurt = float(kurtosis(returns))
    true_kurt = excess_kurt + 3.0

    return {
        "mean": mean,
        "std": std,
        "skew": skewness,
        "excess_kurtosis": excess_kurt,
        "true_kurtosis": true_kurt,
    }


def classify_tail_behavior(true_kurtosis: float) -> str:
    """
    Classify tail behavior from kurtosis.
    """
    if not np.isfinite(true_kurtosis):
        raise ValueError("true_kurtosis must be finite.")

    if true_kurtosis > 3.5:
        return "FAT_TAILED"
    if true_kurtosis < 2.5:
        return "THIN_TAILED"
    return "NEAR_NORMAL"