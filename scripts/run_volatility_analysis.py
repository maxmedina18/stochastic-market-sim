from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.data.returns import log_returns
from market_lab.validation.stylized_facts import (
    autocorrelation,
    rolling_volatility,
    squared_return_autocorrelation,
)


def main() -> None:
    ticker = "EURUSD=X"
    start = "2015-01-01"
    end = "2024-01-01"
    window = 30

    prices = load_adjusted_close_prices(
        ticker=ticker,
        start=start,
        end=end,
    )

    returns = log_returns(prices)
    vol = rolling_volatility(returns, window=window)

    ac1 = autocorrelation(returns, lag=1)
    ac_sq1 = squared_return_autocorrelation(returns, lag=1)

    print("Volatility analysis")
    print(f"Ticker: {ticker}")
    print(f"Observations: {returns.size}")
    print(f"Return autocorrelation (lag 1): {ac1:.6f}")
    print(f"Squared return autocorrelation (lag 1): {ac_sq1:.6f}")
    print(f"Mean rolling volatility: {np.mean(vol):.6f}")
    print(f"Max rolling volatility: {np.max(vol):.6f}")
    print(f"Min rolling volatility: {np.min(vol):.6f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Price series
    axes[0].plot(prices)
    axes[0].set_title(f"{ticker} Price Series")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Price")

    # Plot 2: Log returns
    axes[1].plot(returns)
    axes[1].set_title(f"{ticker} Log Returns")
    axes[1].set_xlabel("Time Index")
    axes[1].set_ylabel("Log Return")

    # Plot 3: Rolling volatility
    axes[2].plot(vol)
    axes[2].set_title(f"{ticker} {window}-Day Rolling Volatility")
    axes[2].set_xlabel("Time Index")
    axes[2].set_ylabel("Volatility")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()