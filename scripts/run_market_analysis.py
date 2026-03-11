from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.data.returns import log_returns


def main():

    ticker = "EURUSD=X"

    prices = load_adjusted_close_prices(
        ticker=ticker,
        start="2015-01-01",
        end="2024-01-01",
    )

    returns = log_returns(prices)

    print("Market analysis: ")
    print(f"Observations: {returns.size}")
    print(f"Mean return: {np.mean(returns):.6f}")
    print(f"Std dev: {np.std(returns):.6f}")
    print(f"Skew: {skew(returns):.4f}")
    print(f"Kurtosis: {kurtosis(returns):.4f}")


if __name__ == "__main__":
    main()