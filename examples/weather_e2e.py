"""End-to-end PoC: tier-1 + tier-2 trainers on synthetic weather-like data.

The synthetic target is a continuous, temperature-like quantity. As in every
example here, we treat it as a prediction-market underlying: model its full
predictive distribution rather than a point estimate, then price a bracket
ladder over it, where each bracket is a YES/NO contract on "the value lands in
this range". This script's job is breadth, exercising most trainers and
backings in one run.

Run:
    conda run -n weathermarkets python -m bracketlearn.examples.weather_e2e

Composition is the native surface: each model is a `Pipeline` (chain) or a
`Stacker` (parallel combiner over upstream objects); the whole list runs under
one `WalkForward` (the CV/OOF driver). Names are leaderboard labels only.

Tier 1 (parametric / mixture backings):
  - ridge:            Pipeline([SklearnPoint(RidgeCV), GlobalResidual])
  - lin_ols:          Pipeline([SklearnPoint(LinearRegression), GlobalResidual]),
                      the same shape as ridge with α=0, written out explicitly.
  - emos:             native parametric-normal DistForecaster
  - emos_calibrated:  Pipeline([EMOS, Isotonic(edges)])
  - ngboost:          non-linear EMOS via NGBoost (native parametric normal)
  - mixture:          per-vendor Gaussian mixture (native parametric mixture)
  - stack:            Stacker([ridge, emos], StackedParametric())

Tier 2 (quantile / bracket backings, conformal calibration, tail specialist):
  - qreg:             LightGBM per-τ quantile heads (quantile-backed)
  - qreg_conformal:   Pipeline([QuantileReg, ConformalCalibrate])
  - qforest:          Random Forest quantile regression (quantile-backed)
  - cumbin:           cumulative-binary classifier (bracket-backed)
  - tail_specialist:  Stacker([emos], TailSpecialist()): EMOS body + LightGBM tails

Tier 3 (online aggregation):
  - online_agg:       sleeping-experts AdaHedge over the K columns of X, lifted
                      to parametric normal via GlobalResidual. (RNNHourly needs
                      3-D X; see weather_rnn_e2e.py.)

PipelineResult.score() owns OOF alignment, so you never touch dist.ids.
"""

from __future__ import annotations

import warnings

import numpy as np

# Quiet the LightGBM "X does not have valid feature names" warning — we
# feed numpy arrays everywhere by design.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

from sklearn.linear_model import LinearRegression

from bracketlearn.compose import Stacker, WalkForward
from bracketlearn.lift import ConformalCalibrate, GlobalResidual
from bracketlearn.pipeline import Pipeline
from bracketlearn.trainers import (
    EMOS,
    CumulativeBinary,
    MixtureNormals,
    NGBoostNormal,
    OnlineAggregator,
    QuantileForest,
    QuantileReg,
    SklearnPoint,
    StackedParametric,
    TailSpecialist,
    emos_calibrated,
    ridge,
)


def make_synthetic_weather(
    n_days: int = 2000,
    n_members: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    truth = 15.0 + 10.0 * np.sin(2 * np.pi * days / 365.0)
    eps = rng.normal(0, 2.0, size=n_days)
    for t in range(1, n_days):
        eps[t] += 0.4 * eps[t - 1]
    y = truth + eps
    member_biases = rng.normal(0, 1.5, size=n_members)
    member_noise = rng.normal(0, 1.8, size=(n_days, n_members))
    X = truth[:, None] + member_biases[None, :] + member_noise
    ids = np.arange(n_days)
    ts = days.astype(float)
    return X, y, ids, ts


def main() -> None:
    print("=" * 70)
    print("bracketlearn v0.1 — tier-1 + tier-2 end-to-end demo")
    print("=" * 70)

    X, y, ids, ts = make_synthetic_weather()
    print(f"data: N={len(y)}, K_members={X.shape[1]}, y range=[{y.min():.1f}, {y.max():.1f}]")

    # Bracket ladder shared between Isotonic, TailSpecialist, CumulativeBinary,
    # and scoring. Width chosen so tail brackets contain observed mass
    # (TailSpecialist refuses to fit with <5 positives per tail).
    edges = np.linspace(3.0, 28.0, 11)   # 10 brackets; outer ones cover ~10% each
    inner_cutpoints = edges[1:-1]        # interior cutpoints for CumulativeBinary

    # v0.3 per-row brackets: the example uses a shared ladder across
    # rows; broadcast it into id-keyed dicts so the trainers can look
    # each row up.
    cutpoints_by_id = {int(i): inner_cutpoints for i in ids}
    outer_edges_by_id = {int(i): (float(edges[0]), float(edges[-1])) for i in ids}
    brackets_by_id = {int(i): edges for i in ids}

    # Tier 1 — names are leaderboard labels; ``ridge()`` / ``emos_calibrated()``
    # already return named Pipelines. ``emos`` is reused by two combiners
    # (stack, tail_specialist) — the SAME object, so it is fit once per fold.
    ridge_node = ridge()
    lin_ols = Pipeline(
        [SklearnPoint(LinearRegression()), GlobalResidual()], name="lin_ols",
    )
    emos = Pipeline([EMOS()], name="emos")
    emos_cal = emos_calibrated(edges=edges)
    ngboost = Pipeline(
        [NGBoostNormal(n_estimators=200, learning_rate=0.02, random_seed=0)],
        name="ngboost",
    )
    mixture = Pipeline([MixtureNormals()], name="mixture")
    stack = Stacker([ridge_node, emos], StackedParametric(), name="stack")
    # Tier 2
    qreg = Pipeline([QuantileReg(n_estimators=100, random_seed=0)], name="qreg")
    qreg_conformal = Pipeline(
        [QuantileReg(n_estimators=100, random_seed=0), ConformalCalibrate()],
        name="qreg_conformal",
    )
    qforest = Pipeline(
        [QuantileForest(n_estimators=200, random_seed=0)], name="qforest",
    )
    cumbin = Pipeline(
        [CumulativeBinary(
            cutpoints_by_id=cutpoints_by_id, outer_edges_by_id=outer_edges_by_id,
        )],
        name="cumbin",
    )
    tail_specialist = Stacker(
        [emos], TailSpecialist(brackets_by_id=brackets_by_id),
        name="tail_specialist",
    )
    # Tier 3
    online_agg = Pipeline(
        [OnlineAggregator(min_experts=2), GlobalResidual()], name="online_agg",
    )

    model = [
        ridge_node, lin_ols, emos, emos_cal, ngboost, mixture, stack,
        qreg, qreg_conformal, qforest, cumbin, tail_specialist, online_agg,
    ]

    print(f"\nfitting (4-fold expanding window, {len(model)} models)...")
    result = WalkForward(
        cv="expanding-window", n_folds=4, embargo=0,
    ).fit_predict(model, X, y, ids=ids, timestamps=ts)
    print(f"got OOF dists for: {result.stages}")

    print("\n[distribution metrics]")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    print(f"\n[bracket metrics — {len(edges) - 1} bins on [{edges[0]:.0f}, {edges[-1]:.0f}]]")
    print(result.to_table(y, metrics=["log_loss_bracket", "brier_bracket"], edges=edges))

    print("\ndone.")


if __name__ == "__main__":
    main()
