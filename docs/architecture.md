# Architecture

The repository is organized around a small number of responsibilities so the mathematical core stays obvious while the codebase remains ready for later growth.

## Top-level folders

- `src/market_lab/`: production research code. This is where mathematically reusable model logic and validation logic live.
- `tests/`: automated checks against the code in `src`. Tests should stay focused on correctness, reproducibility, and analytical consistency.
- `scripts/`: thin entry points for repeatable local runs such as validation sweeps or sanity checks. Scripts should orchestrate code from `src`, not re-implement it.
- `docs/`: short engineering documents that explain the structure and staged direction of the repo.
- `dashboards/`: presentation-facing explainer components and lightweight research visualizations. These stay outside `src` so notebook or UI concerns do not leak into core model code.

## `src/market_lab` layout

- `models/stochastic/`: mathematically explicit stochastic process implementations. GBM lives here because it is a reusable model, not a script concern.
- `validation/`: analytical and empirical checks that test whether models behave as theory predicts.
- `utils/`: reserved for small shared helpers once they are genuinely needed. It is intentionally empty except for package initialization.

## Why this structure

This layout keeps the codebase lean:

- model code is separate from validation and presentation code
- scripts remain thin and disposable
- tests point at importable source code instead of ad hoc notebook logic
- future additions such as pricing, calibration, and backtesting can be added without rewriting the foundation

The existing `stoch/` package remains in place as a compatibility surface for earlier work while the new `market_lab` layout becomes the primary architecture going forward.
