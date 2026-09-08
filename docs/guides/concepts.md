# Concepts

Five protocols, with no inheritance maze.

| Protocol            | Input → Output                                | Examples |
| ------------------- | --------------------------------------------- | -------- |
| `PointForecaster`   | `X → PointForecast` (μ̂)                      | `SklearnPoint(Ridge())`, `OnlineAggregator`, `RNNHourly` |
| `DistForecaster`    | `X → DistributionForecast`                    | `EMOS`, `NGBoostNormal`, `QuantileReg`, `CumulativeBinary` |
| `Lifter`            | `PointForecast → DistributionForecast`        | `GlobalResidual` |
| `Calibrator`        | `DistributionForecast → DistributionForecast` | `Isotonic`, `ConformalCalibrate` |
| `ContractAdapter`   | `DistributionForecast → ContractForecast`     | `BracketLadder` |

List stages in a `Pipeline` and the protocol types wire the chain
left-to-right. A `PointForecaster` followed by a `Lifter` becomes a
`DistForecaster`, and following that with a `Calibrator` leaves it one. The
`Pipeline` is itself the forecaster, as in `Pipeline([SklearnPoint(Ridge()),
GlobalResidual(), Isotonic(...)])`. For parallel ensembling, wrap upstream
`Pipeline` objects in a `Stacker`. `WalkForward` drives the CV and OOF. Names
label the leaderboard, never the wiring.

## Distribution backings

A `DistributionForecast` can carry any of four backings, and metrics dispatch
on the type.

- **parametric** (`normal`, `mixture_normal`) gives closed-form CRPS,
  log-score, and CDF.
- **quantile** is an array of `qvals` at fixed `taus`. CRPS comes from a
  pinball-trapezoidal integral, and the tail policy controls extrapolation
  past the outermost quantile.
- **bracket** is an array of `probs` on `edges` with uniform-within-bin
  density.
- **empirical** (planned) is an array of `members`.

## Provenance

Every `PointForecast` and `DistributionForecast` carries a `ProvenanceMeta`
tag recording which forecaster produced it, which fold, what random seed, and
what conversion chain, for example
`["Ridge", "GlobalResidual", "Isotonic"]`. Lifters and calibrators append
to the conversion chain rather than discarding upstream provenance.
