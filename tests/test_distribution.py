from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.validation.distribution import classify_tail_behavior, distribution_summary


def test_distribution_summary_rejects_non_1d():
    returns = np.array([[0.1, -0.1]])
    with pytest.raises(ValueError, match="one-dimensional"):
        distribution_summary(returns)


def test_distribution_summary_outputs_expected_keys():
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.0])
    summary = distribution_summary(returns)

    assert "mean" in summary
    assert "std" in summary
    assert "skew" in summary
    assert "excess_kurtosis" in summary
    assert "true_kurtosis" in summary


def test_classify_tail_behavior_fat_tailed():
    assert classify_tail_behavior(5.5) == "FAT_TAILED"


def test_classify_tail_behavior_near_normal():
    assert classify_tail_behavior(3.0) == "NEAR_NORMAL"