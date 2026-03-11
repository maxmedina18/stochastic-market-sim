import numpy as np
import matplotlib.pyplot as plt
from market_lab.models.stochastic.gbm import simulate_geometric_brownian_motion

def main():
    T = 1.0
    n_steps = 252
    n_paths = 100
    mu = 0.08
    sigma = 0.2
    S0 = 100.0

    t, S = simulate_geometric_brownian_motion(
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        mu=mu,
        sigma=sigma,
        S0=S0,
        seed=42
    )

    plt.figure(figsize=(10, 6))
    plt.plot(t, S.T, alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Simulated Geometric Brownian Motion Paths")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()