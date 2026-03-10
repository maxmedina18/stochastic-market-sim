from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.validation.brownian_checks import compute_brownian_validation_metrics


def main() -> None:
    metrics = compute_brownian_validation_metrics(
        T=1.0,
        n_steps=252,
        n_paths=100_000,
        seed=42,
    )

    print("Brownian validation")
    print(f"Empirical mean:      {metrics.empirical_mean:.6f}")
    print(f"Theoretical mean:    {metrics.theoretical_mean:.6f}")
    print(f"Mean abs error:      {metrics.mean_absolute_error:.6f}")
    print(f"Empirical variance:  {metrics.empirical_variance:.6f}")
    print(f"Theoretical variance:{metrics.theoretical_variance:.6f}")
    print(f"Variance abs error:  {metrics.variance_absolute_error:.6f}")


if __name__ == "__main__":
    main()