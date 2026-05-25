"""End-to-end PoC: tier-1 trainers on synthetic weather-like data.

Run:
    conda run -n weathermarkets python -m bracketlearn.examples.weather_e2e

Five trainers exercising the major code paths:

  - ridge            — LiftedForecaster(SklearnPoint(RidgeCV) + GlobalResidual)
  - market_ols       — LiftedForecaster(SklearnPoint(LinearRegression) + GlobalResidual)
  - emos             — native parametric-normal DistForecaster
  - emos_calibrated  — CalibratedForecaster(EMOS, Isotonic(edges))
  - ngboost_normal   — non-linear EMOS via NGBoost (native parametric normal)
  - mixture_normals  — per-vendor Gaussian mixture (native parametric mixture)
  - stack            — Stacking meta-learner over (ridge, emos)

PipelineResult.score() owns OOF alignment so the user never touches dist.ids.
"""

from __future__ import annotations

import numpy as np

from bracketlearn.adapters import BracketLadder
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import (
    EMOS,
    MixtureNormals,
    NGBoostNormal,
    Stacking,
    emos_calibrated,
    market_ols,
    ridge,
)


def make_synthetic_weather(
    n_days: int = 500,
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
    print("bracketlearn v0.1 — tier-1 end-to-end demo")
    print("=" * 70)

    X, y, ids, ts = make_synthetic_weather()
    print(f"data: N={len(y)}, K_members={X.shape[1]}, y range=[{y.min():.1f}, {y.max():.1f}]")

    # Bracket ladder shared between Isotonic calibration and contract scoring.
    edges = np.linspace(-10, 40, 11)   # 10 brackets

    pipeline = ForecastPipeline(
        steps=[
            ("ridge",            ridge()),
            ("market_ols",       market_ols()),
            ("emos",             EMOS()),
            ("emos_calibrated",  emos_calibrated(edges=edges)),
            ("ngboost",          NGBoostNormal(n_estimators=200, learning_rate=0.02, random_seed=0)),
            ("mixture",          MixtureNormals()),
            ("stack",            Stacking(deps=("ridge", "emos"))),
        ],
        cv="expanding-window",
        n_folds=5,
        embargo=0,
    )

    print("\nfitting pipeline (5-fold expanding window)...")
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
