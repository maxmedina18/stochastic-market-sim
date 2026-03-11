from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.data.returns import (
    annualized_volatility_from_log_returns,
    log_returns,
    simple_returns,
)


def main() -> None:
    prices = np.array(
        [100.0, 101.2, 100.8, 102.5, 101.7, 103.1, 102.9, 104.4],
        dtype=float,
    )

    sret = simple_returns(prices)
    lret = log_returns(prices)
    ann_vol = annualized_volatility_from_log_returns(lret, periods_per_year=252)

    print("Empirical return validation")
    print(f"Number of prices:         {prices.size}")
    print(f"Number of returns:        {lret.size}")
    print(f"Mean simple return:       {np.mean(sret):.6f}")
    print(f"Mean log return:          {np.mean(lret):.6f}")
    print(f"Std. dev. log return:     {np.std(lret, ddof=1):.6f}")
    print(f"Annualized volatility:    {ann_vol:.6f}")


if __name__ == "__main__":
    main()