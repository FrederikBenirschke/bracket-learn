# bracketlearn

An sklearn-style toolkit for **forecasting a continuous number**, then
**pricing the prediction-market contracts that pay out on it**.

A *prediction market* sells contracts that pay $1 if an event happens and $0
if not — so a contract's price is the market's implied probability (a YES at
31¢ ⇒ a ~31% chance). On Kalshi / Polymarket the same underlying quantity
(today's high temperature, a game's margin, the next GDP print) is sold many
ways: **brackets** (`70–72°F`, `72–74°F`, …), single **thresholds** ("above
75°F"), and **spreads / totals**. To trade them you need your own *calibrated*
distribution over the underlying, and a way to turn that one distribution into
a fair price for every contract shape. bracketlearn is that bridge: **forecast
a distribution → price the contracts → score the prices**, all on one typed
`DistributionForecast`.

The running example in these guides is temperature (the cleanest continuous
underlying), but nothing is weather-specific — any continuous quantity with
bracket / threshold / spread contracts uses the same API.

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
bracketlearn keeps going: every forecast has a typed `DistributionForecast`
that knows how to convert itself onto a bracket ladder and price the
resulting contracts. Calibration, conformal correction, and tail
specialisation are first-class transformer stages — not glue code in your
notebook.

## Install

Pre-PyPI — install from source:

```bash
git clone https://github.com/FrederikBenirschke/bracketlearn
pip install -e ./bracketlearn
pip install -e "./bracketlearn[demo]"   # with optional trainers
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params` / `set_params` / `clone()`. `WalkForward` clones each
model before every fold's fit, so the user-supplied instances are never
mutated and can be safely reused across runs.

## Index

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
