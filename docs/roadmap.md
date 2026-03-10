# Roadmap

## Stage 1: Brownian motion + GBM foundation

- keep stochastic path simulation mathematically transparent
- validate GBM empirically against analytical moments
- establish repeatable tests and sanity-check scripts

## Stage 2: Black-Scholes and Monte Carlo pricing

- add analytical Black-Scholes pricing
- price vanilla options with Monte Carlo
- compare simulation estimates against closed-form benchmarks

## Stage 3: Empirical market validation

- compare model-implied behavior against observed return data
- measure calibration quality and stability
- document where GBM assumptions fail in practice

## Stage 4: Better models (e.g. Heston / Hurst)

- extend beyond constant-volatility dynamics
- evaluate whether richer models improve empirical fit enough to justify complexity
- preserve the same validation-first discipline used for GBM

## Stage 5: Backtesting and risk

- connect validated models to portfolio and strategy workflows
- add backtesting, scenario analysis, and risk reporting
- keep research code and execution-oriented code clearly separated
