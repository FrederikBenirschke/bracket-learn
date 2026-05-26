"""End-to-end PoC: tier-1 + tier-2 trainers on synthetic weather-like data.

Run:
    conda run -n weathermarkets python -m bracketlearn.examples.weather_e2e

Trainers exercising the major code paths:

Tier 1 (parametric / mixture backings):
  - ridge            — LiftedForecaster(SklearnPoint(RidgeCV) + GlobalResidual)
  - market_ols       — LiftedForecaster(SklearnPoint(LinearRegression) + GlobalResidual)
  - emos             — native parametric-normal DistForecaster
  - emos_calibrated  — CalibratedForecaster(EMOS, Isotonic(edges))
  - ngboost          — non-linear EMOS via NGBoost (native parametric normal)
  - mixture          — per-vendor Gaussian mixture (native parametric mixture)
  - stack            — Stacking meta-learner over (ridge, emos)

Tier 2 (quantile / bracket backings, conformal calibration, tail specialist):
  - qreg             — LightGBM per-τ quantile heads (quantile-backed)
  - qreg_conformal   — qreg + ConformalCalibrate (per-τ offsets)
  - qforest          — Random Forest quantile regression (quantile-backed)
  - cumbin           — cumulative-binary classifier (bracket-backed)
  - tail_specialist  — EMOS body + LightGBM tail classifiers (bracket-backed)

Tier 3 (online aggregation):
  - online_agg       — sleeping-experts AdaHedge over the K columns of X,
                       lifted to parametric normal via GlobalResidual.
                       (RNNHourly needs 3-D X — see weather_rnn_e2e.py.)

PipelineResult.score() owns OOF alignment so the user never touches dist.ids.
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

from bracketlearn.adapters import BracketLadder
from bracketlearn.composite import CalibratedForecaster, LiftedForecaster
from bracketlearn.lift import ConformalCalibrate, GlobalResidual
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import (
    EMOS,
    CumulativeBinary,
    MixtureNormals,
    NGBoostNormal,
    OnlineAggregator,
    QuantileForest,
    QuantileReg,
    Stacking,
    TailSpecialist,
    emos_calibrated,
    market_ols,
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

    pipeline = ForecastPipeline(
        steps=[
            # Tier 1
            ("ridge",           ridge()),
            ("market_ols",      market_ols()),
            ("emos",            EMOS()),
            ("emos_calibrated", emos_calibrated(edges=edges)),
            ("ngboost",         NGBoostNormal(n_estimators=200, learning_rate=0.02, random_seed=0)),
            ("mixture",         MixtureNormals()),
            ("stack",           Stacking(deps=("ridge", "emos"))),
            # Tier 2
            ("qreg",            QuantileReg(n_estimators=100, random_seed=0)),
            ("qreg_conformal",  CalibratedForecaster(
                                    QuantileReg(n_estimators=100, random_seed=0),
                                    ConformalCalibrate(),
                                    name="qreg_conformal")),
            ("qforest",         QuantileForest(n_estimators=200, random_seed=0)),
            ("cumbin",          CumulativeBinary(
                                    cutpoints=inner_cutpoints,
                                    outer_edges=(float(edges[0]), float(edges[-1])),
                                )),
            ("tail_specialist", TailSpecialist(edges=edges, upstream="emos")),
            # Tier 3
            ("online_agg",      LiftedForecaster(
                                    base=OnlineAggregator(min_experts=2),
                                    lifter=GlobalResidual(family="normal"),
                                    name="online_agg")),
        ],
        cv="expanding-window",
        n_folds=4,
        embargo=0,
    )

    print(f"\nfitting pipeline (5-fold expanding window, {len(pipeline._stages)} stages)...")
    result = pipeline.fit_predict(X, y, ids=ids, timestamps=ts)
    print(f"got OOF dists for: {result.stages}")

    print("\n[distribution metrics]")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    ladder = BracketLadder(edges=edges)
    print(f"\n[bracket metrics — {len(edges) - 1} bins on [{edges[0]:.0f}, {edges[-1]:.0f}]]")
    print(result.to_table(y, metrics=["log_loss_bracket", "brier_bracket"], ladder=ladder))

    print("\ndone.")


if __name__ == "__main__":
    main()
