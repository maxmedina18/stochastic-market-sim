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

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(17, 12))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.38],
        hspace=0.38,
        wspace=0.28,
    )

    fig.suptitle(f"Spectra Market Structure Summary — {ticker}", fontsize=16)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_returns = fig.add_subplot(gs[1, 0])
    ax_qq = fig.add_subplot(gs[1, 1])
    ax_regimes = fig.add_subplot(gs[2, 0])
    ax_garch = fig.add_subplot(gs[2, 1])
    ax_summary = fig.add_subplot(gs[:, 2])

    # 1. Price series
    ax_price.plot(prices, color="black", linewidth=1.2)
    ax_price.set_title("Price Series")
    ax_price.set_xlabel("Time Index")
    ax_price.set_ylabel("Price")

    # 2. Return distribution + normal overlay
    ax_dist.hist(returns, bins=60, density=True, alpha=0.7)
    x = np.linspace(np.min(returns), np.max(returns), 500)
    y = norm.pdf(x, loc=mu, scale=sigma)
    ax_dist.plot(x, y, linewidth=2)
    ax_dist.set_title("Return Distribution vs Normal")
    ax_dist.set_xlabel("Log Return")
    ax_dist.set_ylabel("Density")

    # 3. Log returns
    ax_returns.plot(returns, linewidth=1.0)
    ax_returns.set_title("Log Returns")
    ax_returns.set_xlabel("Time Index")
    ax_returns.set_ylabel("Return")

    # 4. Q-Q plot
    probplot(returns, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q-Q Plot")

    # 5. Volatility regimes
    ax_regimes.plot(vol_index, realized_vol, color="black", linewidth=1.2, label="Rolling Vol")
    high_vol = regimes == 1
    low_vol = regimes == 0
    ax_regimes.scatter(vol_index[high_vol], realized_vol[high_vol], color="red", s=10, label="High Vol")
    ax_regimes.scatter(vol_index[low_vol], realized_vol[low_vol], color="blue", s=10, label="Low Vol")
    ax_regimes.set_title("Volatility Regimes")
    ax_regimes.set_xlabel("Time Index")
    ax_regimes.set_ylabel("Volatility")
    ax_regimes.legend(loc="upper right", fontsize=9, frameon=True)

    # 6. GARCH vs realized volatility
    ax_garch.plot(garch_vol, color="black", linewidth=1.2, label="GARCH Vol")
    ax_garch.plot(vol_index, realized_vol, color="red", alpha=0.7, label="Realized Vol")
    ax_garch.set_title("GARCH vs Realized Volatility")
    ax_garch.set_xlabel("Time Index")
    ax_garch.set_ylabel("Volatility")
    ax_garch.legend(loc="upper right", fontsize=9, frameon=True)

    # 7. Summary panel
    current_regime = "HIGH_VOL" if regimes[-1] == 1 else "LOW_VOL"

    summary = (
        "Market Diagnostics\n"
        "------------------\n\n"
        f"Hurst exponent: {H:.3f}\n"
        f"Behavior: {hurst_class}\n"
        f"Current regime: {current_regime}\n"
        f"GARCH α+β: {alpha + beta:.2f}\n\n"
        f"Mean return: {mu:.5f}\n"
        f"Std dev: {sigma:.5f}"
    )

    ax_summary.axis("off")
    ax_summary.text(
        0.02,
        0.98,
        summary,
        transform=ax_summary.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="black",
            alpha=0.95,
        ),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()