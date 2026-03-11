from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.models.volatility.garch import (
    garch11_variance,
    garch11_volatility,
)


def test_garch_variance_shape_matches_returns():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    variances = garch11_variance(
        returns=returns,
        omega=1e-6,
        alpha=0.1,
        beta=0.85,
    )
    assert variances.shape == returns.shape


def test_garch_variances_are_positive():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    variances = garch11_variance(
        returns=returns,
        omega=1e-6,
        alpha=0.1,
        beta=0.85,
    )
    assert np.all(variances > 0.0)


def test_garch_volatility_is_sqrt_of_variance():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    variances = garch11_variance(
        returns=returns,
        omega=1e-6,
        alpha=0.1,
        beta=0.85,
    )
    vol = garch11_volatility(
        returns=returns,
        omega=1e-6,
        alpha=0.1,
        beta=0.85,
    )
    np.testing.assert_allclose(vol, np.sqrt(variances))


def test_garch_rejects_non_1d_returns():
    returns = np.array([[0.01, -0.02]])
    with pytest.raises(ValueError, match="one-dimensional"):
        garch11_variance(
            returns=returns,
            omega=1e-6,
            alpha=0.1,
            beta=0.85,
        )


def test_garch_rejects_invalid_stationarity():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    with pytest.raises(ValueError, match="alpha \\+ beta must be less than 1"):
        garch11_variance(
            returns=returns,
            omega=1e-6,
            alpha=0.2,
            beta=0.85,
        )