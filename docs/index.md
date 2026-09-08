# bracketlearn

An sklearn-style toolkit for **forecasting a continuous number**, then
**pricing the prediction-market contracts that pay out on it**.

A *prediction market* sells contracts that pay $1 when an event happens and $0
when it does not, so a contract's price reads as the market's implied
probability, and a YES at 31¢ means a chance near 31%. On Kalshi and Polymarket
the same underlying quantity, such as today's high temperature, a game's margin,
or the next GDP print, sells in several shapes. There are **brackets**
(`70–72°F`, `72–74°F`, …), single **thresholds** ("above 75°F"), and **spreads /
totals**. Trading them requires a *calibrated* distribution over the underlying
and a way to read a fair price for every contract shape off it. bracketlearn
bridges that gap by carrying one typed `DistributionForecast` through three
steps, forecasting a distribution, pricing the contracts, and scoring the
prices.

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
guides/value_vs_accuracy
guides/value_with_fees
guides/value_trainers
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
api/value
api/pool
api/component_lift
api/calibration
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
specialisation run as first-class transformer stages inside the library, in
place of glue code in a notebook.

## Install

Install from source until bracketlearn reaches PyPI.

```bash
git clone https://github.com/FrederikBenirschke/bracket-learn
pip install -e ./bracket-learn
pip install -e "./bracket-learn[demo]"   # with optional trainers
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params`, `set_params`, and `clone()`. `WalkForward` clones each
model before every fold's fit, so the caller's instances stay unmutated and can
be reused across runs.

## Index

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
