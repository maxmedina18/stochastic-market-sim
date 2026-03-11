from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.returns import (
    annualized_volatility_from_log_returns,
    log_returns,
    simple_returns,
    validate_price_series,
)


def test_validate_price_series_rejects_non_1d():
    prices = np.array([[100.0, 101.0]])
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_price_series(prices)


def test_validate_price_series_rejects_short_input():
    prices = np.array([100.0])
    with pytest.raises(ValueError, match="at least two observations"):
        validate_price_series(prices)


def test_validate_price_series_rejects_nonpositive_prices():
    prices = np.array([100.0, 0.0, 101.0])
    with pytest.raises(ValueError, match="positive"):
        validate_price_series(prices)


def test_simple_returns():
    prices = np.array([100.0, 110.0, 121.0])
    returns = simple_returns(prices)
    np.testing.assert_allclose(returns, np.array([0.10, 0.10]))


def test_log_returns():
    prices = np.array([100.0, 110.0])
    returns = log_returns(prices)
    np.testing.assert_allclose(returns, np.array([np.log(1.1)]))


def test_annualized_volatility_is_nonnegative():
    returns = np.array([0.01, -0.02, 0.015, -0.005])
    vol = annualized_volatility_from_log_returns(returns, periods_per_year=252)
    assert vol >= 0.0