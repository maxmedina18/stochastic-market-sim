"""Brownian motion simulation with explicit parameter checks."""

from __future__ import annotations

import numpy as np


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_finite_scalar(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def validate_brownian_parameters(
    T: float,
    n_steps: int,
    n_paths: int,
) -> None:
    """Validate Brownian motion inputs before simulation."""
    _validate_finite_scalar("T", T)
    _validate_positive_integer("n_steps", n_steps)
    _validate_positive_integer("n_paths", n_paths)

    if T <= 0.0:
        raise ValueError("T must be positive.")


def simulate_brownian_motion(
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate standard Brownian motion paths.

    Returns
    -------
    t : np.ndarray
        Time grid with shape (n_steps + 1,).
    paths : np.ndarray
        Brownian motion paths with shape (n_paths, n_steps + 1).

    Notes
    -----
    Standard Brownian motion satisfies:
        W(0) = 0
        W(t + dt) - W(t) ~ N(0, dt)
    """
    validate_brownian_parameters(T=T, n_steps=n_steps, n_paths=n_paths)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    shocks = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    increments = np.sqrt(dt) * shocks

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)

    return t, paths