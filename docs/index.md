# bracketlearn

An sklearn-style toolkit for **forecasting a continuous number**, then
**pricing the prediction-market contracts that pay out on it**.

A *prediction market* sells contracts that pay $1 when an event happens and $0
when it doesn't, so a contract's price reads as the market's implied
probability (a YES at 31¢ means a chance near 31%). On Kalshi and Polymarket
the same underlying quantity (today's high temperature, a game's margin, the
next GDP print) sells many ways: **brackets** (`70–72°F`, `72–74°F`, …), single
**thresholds** ("above 75°F"), and **spreads / totals**. To trade them you need
your own *calibrated* distribution over the underlying and a way to read a fair
price for every contract shape off it. bracketlearn bridges that gap:
**forecast a distribution → price the contracts → score the prices**, all on
one typed `DistributionForecast`.

These guides run on temperature because it makes the cleanest continuous
underlying. Nothing in the library knows about weather. Any continuous quantity
with bracket, threshold, or spread contracts uses the same API.

```{toctree}
:maxdepth: 2
:caption: Guides

guides/quickstart
guides/concepts
guides/catalog
guides/package_map
guides/cv
guides/weights
guides/multitarget
guides/search
guides/persistence
guides/scoring
guides/adapters
guides/baselines
guides/tail_policies
guides/bracket_expander
guides/examples
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/pipeline
api/trainers
api/baselines
api/lift
api/forecast
api/adapters
api/transformers
api/score
api/multitarget
api/search
api/persistence
api/base
```

## Why

Most probabilistic-forecasting libraries stop at "predict a distribution."
bracketlearn carries each forecast further. Every forecast is a typed
`DistributionForecast` that converts itself onto a bracket ladder and prices
the resulting contracts. Calibration, conformal correction, and tail
specialisation run as first-class transformer stages inside the library, where
your notebook used to hold glue code.

## Install

Install from source until bracketlearn reaches PyPI:

```bash
git clone https://github.com/FrederikBenirschke/bracketlearn
pip install -e ./bracketlearn
pip install -e "./bracketlearn[demo]"   # with optional trainers
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params`, `set_params`, and `clone()`. `WalkForward` clones each
model before every fold's fit, so your instances stay unmutated and you reuse
them across runs.

## Index

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
