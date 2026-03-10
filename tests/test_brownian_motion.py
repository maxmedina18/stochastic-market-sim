from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.models.stochastic.brownian_motion import simulate_brownian_motion
from market_lab.validation.brownian_checks import (
    compute_brownian_validation_metrics,
    theoretical_brownian_terminal_mean,
    theoretical_brownian_terminal_variance,
)


def test_brownian_reproducibility():
    _, first = simulate_brownian_motion(T=1.0, n_steps=252, n_paths=8, seed=42)
    _, second = simulate_brownian_motion(T=1.0, n_steps=252, n_paths=8, seed=42)

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"T": 0.0, "n_steps": 10, "n_paths": 10}, "T must be positive."),
        ({"T": 1.0, "n_steps": 0, "n_paths": 10}, "n_steps must be a positive integer."),
        ({"T": 1.0, "n_steps": 10, "n_paths": 0}, "n_paths must be a positive integer."),
    ],
)
def test_brownian_parameter_validation(kwargs, message):
    params = {"T": 1.0, "n_steps": 10, "n_paths": 10}
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        simulate_brownian_motion(**params)


def test_brownian_shapes():
    t, paths = simulate_brownian_motion(T=2.0, n_steps=100, n_paths=5, seed=1)

    assert t.shape == (101,)
    assert paths.shape == (5, 101)


def test_brownian_starts_at_zero():
    _, paths = simulate_brownian_motion(T=1.0, n_steps=50, n_paths=10, seed=1)

    assert np.allclose(paths[:, 0], 0.0)


def test_brownian_terminal_mean_matches_theory():
    metrics = compute_brownian_validation_metrics(
        T=1.0,
        n_steps=252,
        n_paths=100_000,
        seed=42,
    )

    assert metrics.mean_absolute_error < 0.01


def test_brownian_terminal_variance_matches_theory():
    metrics = compute_brownian_validation_metrics(
        T=1.0,
        n_steps=252,
        n_paths=100_000,
        seed=42,
    )

    assert metrics.variance_absolute_error < 0.01


def test_theoretical_brownian_terminal_moments_are_consistent():
    assert theoretical_brownian_terminal_mean() == 0.0
    assert theoretical_brownian_terminal_variance(T=0.75) == pytest.approx(0.75)