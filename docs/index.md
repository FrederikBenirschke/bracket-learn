# bracketlearn

Sklearn-style framework for **probabilistic forecasting** + **bracket-contract
pricing**.

Built for the case where you predict a continuous quantity (temperature,
score margin, asset return) and need to price a ladder of binary contracts
("HIGH > 75°F?", "score in [10, 20)?") against the forecast distribution.

```{toctree}
:maxdepth: 2
:caption: Guides

guides/quickstart
guides/concepts
guides/cv
guides/weights
guides/multitarget
guides/search
guides/persistence
guides/examples
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/pipeline
api/trainers
api/composite
api/lift
api/forecast
api/adapters
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

```bash
pip install bracketlearn

# With the full set of optional trainers (LightGBM, NGBoost, torch, ...):
pip install "bracketlearn[demo]"
```

## sklearn contract

Every forecaster, lifter, and calibrator inherits from `BaseEstimator` and
supports `get_params` / `set_params` / `clone()`. The pipeline clones each
stage's forecaster before every fold's fit, so the user-supplied
instances are never mutated and can be safely reused across pipelines.

## Index

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
