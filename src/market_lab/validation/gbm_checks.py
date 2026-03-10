"""Reusable analytical checks for geometric Brownian motion simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_lab.models.stochastic.gbm import simulate_geometric_brownian_motion


@dataclass(frozen=True)
class GBMValidationMetrics:
    empirical_mean: float
    theoretical_mean: float
    empirical_variance: float
    theoretical_variance: float
    mean_absolute_error: float
    variance_absolute_error: float


def theoretical_gbm_terminal_mean(S0: float, mu: float, T: float) -> float:
    return float(S0 * np.exp(mu * T))


def theoretical_gbm_terminal_variance(S0: float, mu: float, sigma: float, T: float) -> float:
    growth = np.exp(2.0 * mu * T)
    diffusion = np.exp(sigma**2 * T) - 1.0
    return float((S0**2) * growth * diffusion)


def compute_gbm_validation_metrics(
    T: float,
    n_steps: int,
    n_paths: int,
    mu: float,
    sigma: float,
    S0: float,
    seed: int | None = None,
) -> GBMValidationMetrics:
    _, paths = simulate_geometric_brownian_motion(
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        mu=mu,
        sigma=sigma,
        S0=S0,
        seed=seed,
    )
    terminal_values = paths[:, -1]
    empirical_mean = float(np.mean(terminal_values))
    empirical_variance = float(np.var(terminal_values))
    theoretical_mean = theoretical_gbm_terminal_mean(S0=S0, mu=mu, T=T)
    theoretical_variance = theoretical_gbm_terminal_variance(S0=S0, mu=mu, sigma=sigma, T=T)
    return GBMValidationMetrics(
        empirical_mean=empirical_mean,
        theoretical_mean=theoretical_mean,
        empirical_variance=empirical_variance,
        theoretical_variance=theoretical_variance,
        mean_absolute_error=abs(empirical_mean - theoretical_mean),
        variance_absolute_error=abs(empirical_variance - theoretical_variance),
    )
