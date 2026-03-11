import numpy as np
from market_lab.models.stochastic.brownian_motion import simulate_brownian_motion
from market_lab.models.stochastic.gbm import simulate_geometric_brownian_motion


def test_brownian_motion_statistics():

    T = 1.0
    n_steps = 252
    n_paths = 100000
    t, W = simulate_brownian_motion(T, n_steps, n_paths, seed=42)
    WT = W[:, -1]  # final values at time T
    mean_est = np.mean(WT)
    var_est = np.var(WT)
    print("Estimated mean:", mean_est)
    print("Estimated variance:", var_est)
    assert abs(mean_est) < 0.01
    assert abs(var_est - T) < 0.02

def test_gbm_mean():
    T = 1.0
    n_steps = 252
    n_paths = 100000
    mu = 0.05
    sigma = 0.2
    S0 = 1.0
    t, S = simulate_geometric_brownian_motion(T, n_steps, n_paths, mu, sigma, S0, seed=42)
    ST = S[:, -1]
    empirical_mean = np.mean(ST)
    theoretical_mean = S0 * np.exp(mu * T)
    assert abs(empirical_mean - theoretical_mean) < 0.01