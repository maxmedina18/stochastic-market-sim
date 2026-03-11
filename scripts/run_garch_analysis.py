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
from market_lab.models.volatility.garch import garch11_volatility
from market_lab.validation.stylized_facts import rolling_volatility


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

    # Simple hand-chosen parameters for first pass
    omega = 1e-6
    alpha = 0.10
    beta = 0.85

    garch_vol = garch11_volatility(
        returns=returns,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    realized_vol = rolling_volatility(returns, window=window)
    realized_index = np.arange(window - 1, window - 1 + len(realized_vol))
    garch_index = np.arange(len(garch_vol))

    print("GARCH(1,1) analysis")
    print(f"Ticker: {ticker}")
    print(f"Observations: {returns.size}")
    print(f"omega: {omega}")
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")
    print(f"alpha + beta: {alpha + beta:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].plot(returns)
    axes[0].set_title(f"{ticker} Log Returns")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Log Return")

    axes[1].plot(garch_index, garch_vol, color="black", linewidth=1.5, label="GARCH Forecast Vol")
    axes[1].plot(realized_index, realized_vol, color="red", alpha=0.7, label=f"{window}-Day Realized Vol")
    axes[1].set_title(f"{ticker} GARCH(1,1) Volatility vs Realized Volatility")
    axes[1].set_xlabel("Time Index")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()