from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, norm, probplot, skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.data.returns import log_returns


def main() -> None:
    ticker = "EURUSD=X"
    start = "2015-01-01"
    end = "2024-01-01"

    prices = load_adjusted_close_prices(
        ticker=ticker,
        start=start,
        end=end,
    )

    returns = log_returns(prices)

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skewness = skew(returns)
    excess_kurt = kurtosis(returns)
    true_kurt = excess_kurt + 3.0

    print("Return distribution analysis")
    print(f"Ticker: {ticker}")
    print(f"Observations: {returns.size}")
    print(f"Mean return: {mu:.6f}")
    print(f"Std dev: {sigma:.6f}")
    print(f"Skew: {skewness:.6f}")
    print(f"Excess kurtosis: {excess_kurt:.6f}")
    print(f"True kurtosis: {true_kurt:.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram + normal overlay
    axes[0].hist(returns, bins=60, density=True, alpha=0.7)
    x = np.linspace(np.min(returns), np.max(returns), 500)
    y = norm.pdf(x, loc=mu, scale=sigma)
    axes[0].plot(x, y, linewidth=2)
    axes[0].set_title(f"{ticker} Log Return Distribution vs Normal")
    axes[0].set_xlabel("Log Return")
    axes[0].set_ylabel("Density")

    # Q-Q plot
    probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title(f"{ticker} Normal Q-Q Plot")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()