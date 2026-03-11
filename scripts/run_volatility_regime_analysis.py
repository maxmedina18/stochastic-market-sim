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
from market_lab.validation.stylized_facts import rolling_volatility
from market_lab.validation.volatility_regime import detect_volatility_regimes


def main():

    ticker = "EURUSD=X"
    start = "2015-01-01"
    end = "2024-01-01"
    window = 30

    prices = load_adjusted_close_prices(ticker, start, end)
    returns = log_returns(prices)

    vol = rolling_volatility(returns, window)
    regimes = detect_volatility_regimes(vol)

    vol_index = np.arange(window - 1, window - 1 + len(vol))

    fig, ax = plt.subplots(figsize=(12, 5))

    # main volatility line
    ax.plot(
        vol_index,
        vol,
        color="black",
        linewidth=1.5,
        label="Rolling Volatility",
    )

    high_vol = regimes == 1
    low_vol = regimes == 0

    # highlight regimes
    ax.scatter(
        vol_index[high_vol],
        vol[high_vol],
        color="red",
        s=15,
        label="High Vol Regime",
    )

    ax.scatter(
        vol_index[low_vol],
        vol[low_vol],
        color="blue",
        s=15,
        label="Low Vol Regime",
    )

    ax.set_title(f"{ticker} Volatility Regime Detection")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Volatility")

    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()