"""Market data loader using Yahoo Finance."""

from __future__ import annotations

import numpy as np
import yfinance as yf


def load_adjusted_close_prices(
    ticker: str,
    start: str,
    end: str,
) -> np.ndarray:
    """
    Load adjusted close prices from Yahoo Finance.

    Falls back to Close if Adj Close is unavailable.
    Always returns a one-dimensional NumPy array.
    """
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError("No market data downloaded.")

    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise ValueError("Close prices not found in downloaded data.")

    prices = np.asarray(prices, dtype=float).reshape(-1)

    if prices.size < 2:
        raise ValueError("Not enough price data downloaded.")
    if not np.all(np.isfinite(prices)):
        raise ValueError("Downloaded prices contain non-finite values.")

    return prices