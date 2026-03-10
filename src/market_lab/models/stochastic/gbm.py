"""Geometric Brownian motion simulation with explicit parameter checks."""

from __future__ import annotations

import numpy as np


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_finite_scalar(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def validate_gbm_parameters(
    T: float,
    n_steps: int,
    n_paths: int,
    mu: float,
    sigma: float,
    S0: float,
) -> None:
    """Validate GBM inputs before simulation."""
    _validate_finite_scalar("T", T)
    _validate_positive_integer("n_steps", n_steps)
    _validate_positive_integer("n_paths", n_paths)
    _validate_finite_scalar("mu", mu)
    _validate_finite_scalar("sigma", sigma)
    _validate_finite_scalar("S0", S0)

    if T <= 0.0:
        raise ValueError("T must be positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if S0 <= 0.0:
        raise ValueError("S0 must be positive.")


def simulate_geometric_brownian_motion(
    T: float,
    n_steps: int,
    n_paths: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    S0: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate paths from the exact discretization of geometric Brownian motion.

    The model is
        dS_t = mu S_t dt + sigma S_t dW_t
    with closed-form solution
        S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t).
    """
    validate_gbm_parameters(T=T, n_steps=n_steps, n_paths=n_paths, mu=mu, sigma=sigma, S0=S0)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    shocks = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 1:] = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)
    return t, paths
