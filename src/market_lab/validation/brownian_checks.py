"""Reusable analytical checks for Brownian motion simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_lab.models.stochastic.brownian_motion import simulate_brownian_motion


@dataclass(frozen=True)
class BrownianValidationMetrics:
    empirical_mean: float
    theoretical_mean: float
    empirical_variance: float
    theoretical_variance: float
    mean_absolute_error: float
    variance_absolute_error: float


def theoretical_brownian_terminal_mean() -> float:
    return 0.0


def theoretical_brownian_terminal_variance(T: float) -> float:
    return float(T)


def compute_brownian_validation_metrics(
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> BrownianValidationMetrics:
    _, paths = simulate_brownian_motion(
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    terminal_values = paths[:, -1]
    empirical_mean = float(np.mean(terminal_values))
    empirical_variance = float(np.var(terminal_values))

    theoretical_mean = theoretical_brownian_terminal_mean()
    theoretical_variance = theoretical_brownian_terminal_variance(T=T)

    return BrownianValidationMetrics(
        empirical_mean=empirical_mean,
        theoretical_mean=theoretical_mean,
        empirical_variance=empirical_variance,
        theoretical_variance=theoretical_variance,
        mean_absolute_error=abs(empirical_mean - theoretical_mean),
        variance_absolute_error=abs(empirical_variance - theoretical_variance),
    )