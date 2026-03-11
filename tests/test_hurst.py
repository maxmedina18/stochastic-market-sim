from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.validation.hurst import classify_hurst, hurst_exponent


def test_hurst_rejects_non_1d_input():
    series = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="one-dimensional"):
        hurst_exponent(series)


def test_hurst_rejects_short_series():
    series = np.linspace(0.0, 1.0, 20)
    with pytest.raises(ValueError, match="too short"):
        hurst_exponent(series, max_lag=50)


def test_classify_hurst_trending():
    assert classify_hurst(0.61) == "TRENDING"


def test_classify_hurst_mean_reverting():
    assert classify_hurst(0.39) == "MEAN_REVERTING"


def test_classify_hurst_random_walk():
    assert classify_hurst(0.50) == "RANDOM_WALK"


def test_hurst_random_walk_like_series_is_reasonable():
    rng = np.random.default_rng(42)
    increments = rng.normal(0.0, 1.0, size=5000)
    series = np.cumsum(increments)

    H = hurst_exponent(series, max_lag=100)

    assert 0.35 < H < 0.65