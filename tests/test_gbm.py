from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.models.stochastic.gbm import simulate_geometric_brownian_motion
from market_lab.validation.gbm_checks import (
    compute_gbm_validation_metrics,
    theoretical_gbm_terminal_mean,
    theoretical_gbm_terminal_variance,
)


def test_gbm_reproducibility():
    _, first = simulate_geometric_brownian_motion(
        T=1.0,
        n_steps=252,
        n_paths=8,
        mu=0.05,
        sigma=0.2,
        S0=1.0,
        seed=42,
    )
    _, second = simulate_geometric_brownian_motion(
        T=1.0,
        n_steps=252,
        n_paths=8,
        mu=0.05,
        sigma=0.2,
        S0=1.0,
        seed=42,
    )

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"T": 0.0, "n_steps": 10, "n_paths": 10}, "T must be positive."),
        ({"T": 1.0, "n_steps": 0, "n_paths": 10}, "n_steps must be a positive integer."),
        ({"T": 1.0, "n_steps": 10, "n_paths": 0}, "n_paths must be a positive integer."),
        ({"T": 1.0, "n_steps": 10, "n_paths": 10, "sigma": -0.1}, "sigma must be non-negative."),
        ({"T": 1.0, "n_steps": 10, "n_paths": 10, "S0": 0.0}, "S0 must be positive."),
    ],
)
def test_gbm_parameter_validation(kwargs, message):
    params = {"T": 1.0, "n_steps": 10, "n_paths": 10, "mu": 0.05, "sigma": 0.2, "S0": 1.0}
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        simulate_geometric_brownian_motion(**params)


def test_gbm_terminal_mean_matches_theory():
    metrics = compute_gbm_validation_metrics(
        T=1.0,
        n_steps=252,
        n_paths=100_000,
        mu=0.05,
        sigma=0.2,
        S0=1.0,
        seed=42,
    )

    assert metrics.mean_absolute_error < 0.01


def test_gbm_terminal_variance_matches_theory():
    metrics = compute_gbm_validation_metrics(
        T=1.0,
        n_steps=252,
        n_paths=100_000,
        mu=0.05,
        sigma=0.2,
        S0=1.0,
        seed=42,
    )

    assert metrics.variance_absolute_error < 0.01


def test_theoretical_terminal_moments_are_consistent():
    mean = theoretical_gbm_terminal_mean(S0=2.0, mu=0.1, T=0.5)
    variance = theoretical_gbm_terminal_variance(S0=2.0, mu=0.1, sigma=0.3, T=0.5)

    assert mean == pytest.approx(2.0 * np.exp(0.05))
    assert variance == pytest.approx((2.0**2) * np.exp(0.1) * (np.exp(0.045) - 1.0))

def test_gbm_starts_at_initial_price():
    S0 = 3.5
    _, paths = simulate_geometric_brownian_motion(
        T=1.0,
        n_steps=50,
        n_paths=10,
        mu=0.05,
        sigma=0.2,
        S0=S0,
        seed=1,
    )

    assert np.allclose(paths[:, 0], S0)


def test_gbm_paths_remain_positive():
    _, paths = simulate_geometric_brownian_motion(
        T=1.0,
        n_steps=252,
        n_paths=100,
        mu=0.05,
        sigma=0.2,
        S0=1.0,
        seed=1,
    )

    assert np.all(paths > 0.0)