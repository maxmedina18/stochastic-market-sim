from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pathlib import Path
from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.data.returns import log_returns
from market_lab.models.volatility.garch import garch11_volatility
from market_lab.validation.distribution import classify_tail_behavior, distribution_summary
from market_lab.validation.hurst import classify_hurst, hurst_exponent
from market_lab.validation.regime_summary import current_regime_label
from market_lab.validation.stylized_facts import (
    autocorrelation,
    rolling_volatility,
    squared_return_autocorrelation,
)
from market_lab.validation.volatility_regime import detect_volatility_regimes


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

    # Distribution diagnostics
    dist = distribution_summary(returns)
    tail_class = classify_tail_behavior(dist["true_kurtosis"])

    # Serial dependence diagnostics
    ret_ac1 = autocorrelation(returns, lag=1)
    sq_ac1 = squared_return_autocorrelation(returns, lag=1)

    # Hurst / market structure
    H = hurst_exponent(prices, max_lag=100)
    hurst_class = classify_hurst(H)

    # Volatility regime
    realized_vol = rolling_volatility(returns, window=window)
    regimes = detect_volatility_regimes(realized_vol, threshold_quantile=0.75)
    current_regime = current_regime_label(regimes)

    # GARCH model
    omega = 1e-6
    alpha = 0.10
    beta = 0.85
    garch_vol = garch11_volatility(
        returns=returns,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    current_garch_vol = float(garch_vol[-1])
    current_realized_vol = float(realized_vol[-1])

    print("Market Structure Report")
    print("-----------------------")
    print(f"Ticker: {ticker}")
    print(f"Observations: {returns.size}")
    print()

    print("Distribution")
    print(f"Mean return:            {dist['mean']:.6f}")
    print(f"Std dev:                {dist['std']:.6f}")
    print(f"Skew:                   {dist['skew']:.6f}")
    print(f"True kurtosis:          {dist['true_kurtosis']:.6f}")
    print(f"Tail classification:    {tail_class}")
    print()

    print("Dependence Structure")
    print(f"Return autocorr (lag1): {ret_ac1:.6f}")
    print(f"Squared return ac1:     {sq_ac1:.6f}")
    print()

    print("Market Structure")
    print(f"Hurst exponent:         {H:.6f}")
    print(f"Behavior:               {hurst_class}")
    print()

    print("Volatility")
    print(f"Current regime:         {current_regime}")
    print(f"Current realized vol:   {current_realized_vol:.6f}")
    print(f"Current GARCH vol:      {current_garch_vol:.6f}")
    print(f"GARCH alpha + beta:     {alpha + beta:.4f}")
    print()

    print("Interpretation")
    if hurst_class == "TRENDING":
        print("- Price dynamics appear persistent / trend-following.")
    elif hurst_class == "MEAN_REVERTING":
        print("- Price dynamics show mean-reverting tendency.")
    else:
        print("- Price dynamics are broadly consistent with a random walk.")

    if tail_class == "FAT_TAILED":
        print("- Extreme moves occur more often than a normal model predicts.")

    if sq_ac1 > 0.05:
        print("- Volatility clustering is present.")
    else:
        print("- Volatility clustering appears weak.")

    if current_regime == "HIGH_VOL":
        print("- Market is currently in a high-volatility regime.")
    else:
        print("- Market is currently in a low-volatility regime.")


if __name__ == "__main__":
    main()