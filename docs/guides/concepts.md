# Concepts

Five protocols, no inheritance maze:

| Protocol            | Input → Output                                | Examples |
| ------------------- | --------------------------------------------- | -------- |
| `PointForecaster`   | `X → PointForecast` (μ̂)                      | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly` |
| `DistForecaster`    | `X → DistributionForecast`                    | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary` |
| `Lifter`            | `PointForecast → DistributionForecast`        | `GlobalResidual` |
| `Calibrator`        | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate` |
| `ContractAdapter`   | `DistributionForecast → ContractForecast`     | `BracketLadder` |

Compose stages by listing them in a `Pipeline`: a `PointForecaster`
followed by a `Lifter` becomes a `DistForecaster`, and a `DistForecaster`
followed by a `Calibrator` stays a `DistForecaster`. The chain is wired
left→right by protocol type — `Pipeline([SklearnPoint(Ridge()),
GlobalResidual(), Isotonic(...)])` — and *is* the forecaster. Parallel
ensembling is a `Stacker` over upstream `Pipeline` objects; `WalkForward`
drives the CV/OOF. Names are leaderboard labels, never wiring.

## Distribution backings

A `DistributionForecast` can carry any of four backings; metrics dispatch
on the type:

- **parametric** (`normal`, `mixture_normal`) — closed-form CRPS /
  log-score / CDF.
- **quantile** — array of `qvals` at fixed `taus`; CRPS via pinball
  trapezoidal integral; tail policy controls extrapolation beyond
  outermost quantile.
- **bracket** — array of `probs` on `edges`; uniform-within-bin density.
- **empirical** (planned) — array of `members`.

## Provenance

Every `PointForecast` and `DistributionForecast` carries a
`ProvenanceMeta` tag: which forecaster produced it, which fold, what
random seed, what conversion chain (e.g.
`["Ridge", "GlobalResidual", "Isotonic"]`). Lifters and calibrators append
to the conversion chain rather than discarding upstream provenance.
