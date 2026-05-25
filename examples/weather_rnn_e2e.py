"""RNN-on-hourly-tensor end-to-end PoC.

Standalone because RNNHourly requires X.ndim == 3 (the (N, T, C) hourly
tensor convention) and the main weather_e2e demo runs on 2-D feature
matrices. Pipeline slicing on axis 0 works identically — the only
restriction is that you can't mix 2-D and 3-D trainers in one pipeline.

Run::

    conda run -n weathermarkets python -m bracketlearn.examples.weather_rnn_e2e

Trainers:
  - rnn_hourly  — GRU(32) on the 24-hour, 6-channel tensor + station
                  embedding, lifted to parametric normal via
                  GlobalResidual.

Synthetic data mirrors the real HRRR hourly tensor shape (N, 24, 6):
channels = (temperature_f, dewpoint_f, RH, wind, cloud, CAPE). Target
= daily HIGH ≈ max(T) − 0.3·mean(cloud) + warm-season term + noise.

The RNN learns the cloud-correction residual; ridge can't because it
sees a flat 6-D mean and misses the cloud signal.
"""

from __future__ import annotations

import os

# Set before torch import (per Rule #0 in rnn_hourly.py): macOS libomp clash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from bracketlearn.composite import LiftedForecaster
from bracketlearn.lift import GlobalResidual
from bracketlearn.pipeline import ForecastPipeline
from bracketlearn.trainers import RNNHourly


def make_synthetic_hourly(
    n_days: int = 600,
    n_hours: int = 24,
    n_channels: int = 6,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    seasonal = 15.0 + 10.0 * np.sin(2 * np.pi * days / 365.0)
    X = np.zeros((n_days, n_hours, n_channels), dtype=np.float32)
    hour_of_day = np.arange(n_hours)
    diurnal = 8.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
    for d in range(n_days):
        # ch 0: temperature_f
        X[d, :, 0] = seasonal[d] + diurnal + rng.normal(0, 1.5, n_hours)
        # ch 1: dewpoint_f
        X[d, :, 1] = X[d, :, 0] - rng.uniform(5, 20)
        # ch 2: relative_humidity — anti-correlated with T
        X[d, :, 2] = np.clip(80 - 1.5 * (X[d, :, 0] - seasonal[d]), 0, 100)
        # ch 3: wind, ch 4: cloud, ch 5: CAPE — random.
        X[d, :, 3] = rng.gamma(2, 3, n_hours)
        X[d, :, 4] = rng.uniform(0, 100, n_hours)
        X[d, :, 5] = rng.gamma(1.5, 100, n_hours)
    # Target: realized HIGH = max(T) − 0.3·mean(cloud) + noise.
    baseline = X[:, :, 0].max(axis=1)
    y = baseline - 0.3 * X[:, :, 4].mean(axis=1) + rng.normal(0, 1.2, n_days)
    ids = np.arange(n_days)
    ts = days.astype(float)
    return X, y, ids, ts


def main() -> None:
    print("=" * 70)
    print("bracketlearn v0.1 — tier-3 RNN-on-hourly-tensor demo")
    print("=" * 70)

    X, y, ids, ts = make_synthetic_hourly()
    print(f"data: N={len(y)}, tensor shape (N, T, C) = {X.shape}")
    baseline = X[:, :, 0].max(axis=1)
    print(f"baseline (channel-0 max) MAE vs y: {np.mean(np.abs(baseline - y)):.2f}")

    pipeline = ForecastPipeline(
        steps=[
            ("rnn_hourly", LiftedForecaster(
                base=RNNHourly(epochs=40, hidden=24, embed=2, dropout=0.1),
                lifter=GlobalResidual(family="normal"),
                name="rnn_hourly",
            )),
        ],
        cv="expanding-window",
        n_folds=3,
        embargo=0,
    )

    print("\nfitting pipeline (3-fold expanding window, 3-D X)...")
    result = pipeline.fit_predict(X, y, ids=ids, timestamps=ts)
    print(f"got OOF dists for: {result.stages}")

    print("\n[distribution metrics]")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    rnn_dist = result["rnn_hourly"]
    y_oof = y[rnn_dist.ids.astype(int)]
    rnn_mae = float(np.mean(np.abs(rnn_dist.params["mu"] - y_oof)))
    base_mae = float(np.mean(np.abs(baseline[rnn_dist.ids.astype(int)] - y_oof)))
    print(f"\nRNN MAE: {rnn_mae:.2f}    baseline MAE: {base_mae:.2f}    "
          f"Δ: {base_mae - rnn_mae:+.2f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
