from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, probplot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.loaders.yfinance_loader import load_adjusted_close_prices
from market_lab.data.returns import log_returns
from market_lab.models.volatility.garch import garch11_volatility
from market_lab.validation.hurst import classify_hurst, hurst_exponent
from market_lab.validation.stylized_facts import rolling_volatility
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

    # Distribution stats
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Volatility
    realized_vol = rolling_volatility(returns, window=window)
    regimes = detect_volatility_regimes(realized_vol, threshold_quantile=0.75)
    vol_index = np.arange(window - 1, window - 1 + len(realized_vol))

    # GARCH
    omega = 1e-6
    alpha = 0.10
    beta = 0.85
    garch_vol = garch11_volatility(
        returns=returns,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )

    # Hurst
    H = hurst_exponent(prices, max_lag=100)
    hurst_class = classify_hurst(H)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Spectra Market Structure Summary — {ticker}", fontsize=16)

    # 1. Price series
    axes[0, 0].plot(prices, color="black", linewidth=1.2)
    axes[0, 0].set_title("Price Series")
    axes[0, 0].set_xlabel("Time Index")
    axes[0, 0].set_ylabel("Price")

    # 2. Return distribution + normal overlay
    axes[0, 1].hist(returns, bins=60, density=True, alpha=0.7)
    x = np.linspace(np.min(returns), np.max(returns), 500)
    y = norm.pdf(x, loc=mu, scale=sigma)
    axes[0, 1].plot(x, y, linewidth=2)
    axes[0, 1].set_title("Return Distribution vs Normal")
    axes[0, 1].set_xlabel("Log Return")
    axes[0, 1].set_ylabel("Density")

    # 3. Log returns
    axes[1, 0].plot(returns, linewidth=1.0)
    axes[1, 0].set_title("Log Returns")
    axes[1, 0].set_xlabel("Time Index")
    axes[1, 0].set_ylabel("Return")

    # 4. Q-Q plot
    probplot(returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q Plot")

    # 5. Volatility regimes
    axes[2, 0].plot(vol_index, realized_vol, color="black", linewidth=1.2, label="Rolling Vol")
    high_vol = regimes == 1
    low_vol = regimes == 0
    axes[2, 0].scatter(vol_index[high_vol], realized_vol[high_vol], color="red", s=10, label="High Vol")
    axes[2, 0].scatter(vol_index[low_vol], realized_vol[low_vol], color="blue", s=10, label="Low Vol")
    axes[2, 0].set_title("Volatility Regimes")
    axes[2, 0].set_xlabel("Time Index")
    axes[2, 0].set_ylabel("Volatility")
    axes[2, 0].legend()

    # 6. GARCH vs realized volatility
    axes[2, 1].plot(garch_vol, color="black", linewidth=1.2, label="GARCH Vol")
    axes[2, 1].plot(vol_index, realized_vol, color="red", alpha=0.7, label="Realized Vol")
    axes[2, 1].set_title("GARCH vs Realized Volatility")
    axes[2, 1].set_xlabel("Time Index")
    axes[2, 1].set_ylabel("Volatility")
    axes[2, 1].legend()

    # Add summary text box
    summary = (
        f"Hurst exponent: {H:.3f}\n"
        f"Behavior: {hurst_class}\n"
        f"Current regime: {'HIGH_VOL' if regimes[-1] == 1 else 'LOW_VOL'}\n"
        f"GARCH α+β: {alpha + beta:.2f}"
    )

    fig.text(
        0.78,
        0.92,
        summary,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )
    plt.savefig(
    "spectra_market_structure.png",
    dpi=300,
    bbox_inches="tight"
)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()