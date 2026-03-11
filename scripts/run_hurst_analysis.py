from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.validation.hurst import classify_hurst, hurst_exponent


def main() -> None:
    ticker = "EURUSD=X"
    start = "2015-01-01"
    end = "2024-01-01"

    prices = load_adjusted_close_prices(
        ticker=ticker,
        start=start,
        end=end,
    )

    H = hurst_exponent(prices, max_lag=100)
    classification = classify_hurst(H)

    print("Market Structure Analysis")
    print("-------------------------")
    print(f"Ticker: {ticker}")
    print(f"Hurst exponent: {H:.6f}")
    print(f"Market behavior: {classification}")


if __name__ == "__main__":
    main()