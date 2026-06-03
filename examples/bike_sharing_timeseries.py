"""Hourly bike-sharing demand → time-series probabilistic forecast.

Dataset: OpenML "Bike_Sharing_Demand" (17 379 hourly rows from a DC bike-share
system, 2011–2012). Target = hourly rental count. Numeric features:
``month, hour, weekday, workingday, temp, feel_temp, humidity, windspeed``
plus one-hot ``season`` and ``weather``.

Run::

    conda run -n weathermarkets python -m bracketlearn.examples.bike_sharing_timeseries

What this script demonstrates:

- ``cv="expanding-window"`` on a genuine time series — train always
  precedes test in calendar time, no look-ahead.
- ``Pipeline([EMOS(), Isotonic(...)])`` — EMOS with isotonic calibration,
  fit per fold on a held-out tail of the train window.
- A bracket ladder over realistic demand levels (0 → 1000 bikes/hour).
- Demonstrates the user does NOT touch ``dist.ids`` — ``result.score(y)``
  aligns OOF coverage internally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names.*",
    category=UserWarning,
)

from bracketlearn.adapters import BracketLadder
from bracketlearn.baselines import EmpiricalDistribution, Persistence
from bracketlearn.compose import WalkForward
from bracketlearn.lift import GlobalResidual, Isotonic
from bracketlearn.pipeline import Pipeline
from bracketlearn.trainers import EMOS, QuantileReg


def _prepare(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode every categorical column; return a dense float matrix.

    Keeping it ~simple/single-file: pandas get_dummies, then to_numpy.
    """
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    dummies = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    X = pd.concat([df[num_cols].astype(float), dummies], axis=1)
    return X.to_numpy(dtype=float)


def main() -> None:
    print("loading Bike_Sharing_Demand from OpenML (cached after first run) …")
    ds = fetch_openml("Bike_Sharing_Demand", version=2,
                      as_frame=True, parser="pandas")
    df: pd.DataFrame = ds.data
    y_raw = ds.target.to_numpy(dtype=float)
    # Sort by (year, month, hour) so the row index is the time index. The
    # dataset is already in order but we make it explicit.
    df = df.sort_values(["year", "month", "hour"]).reset_index(drop=True)
    y = y_raw[df.index.to_numpy()]
    X = _prepare(df)
    n = X.shape[0]
    ids = np.arange(n)
    ts = ids.astype(float)        # monotone synthetic timestamp == row order
    print(f"  rows={n}  features={X.shape[1]}  y in [{y.min():.0f}, {y.max():.0f}]")

    # EMOS needs an "ensemble"-shaped X (rows × experts). We synthesise
    # three "vendor" forecasts by perturbing the temperature column, so
    # EMOS has something to mean/var over. This is contrived but lets us
    # show EMOS on a non-weather dataset; in real use you'd plug ensemble
    # columns from your forecast vendors.
    rng = np.random.default_rng(0)
    temp_col = df["temp"].to_numpy(dtype=float)
    X_emos = np.column_stack([
        temp_col + rng.normal(0, 1, n),
        temp_col + rng.normal(0, 2, n),
        temp_col + rng.normal(0, 0.5, n),
    ]) * 8.0 + 50.0   # rough scaling toward ride-count units

    # Bracket ladder spanning the observed range: 0 → 1000 in 100-bike steps.
    edges = np.array([0., 50., 100., 200., 350., 500., 750., 1000.])
    ladder = BracketLadder(edges=edges)
    print(f"ladder: {len(edges)-1} brackets covering {edges[0]:.0f}–{edges[-1]:.0f} bikes/hour")

    print("\nfitting (expanding-window, 5 folds) …")
    model = [
        # Baseline 1: marginal-y distribution, ignores everything.
        Pipeline([EmpiricalDistribution()], name="emp"),
        # Baseline 2: same hour yesterday + global residual σ.
        # On hourly bike demand the diurnal cycle is the dominant
        # signal, so lag-24 persistence is a non-trivial bar to clear.
        Pipeline([Persistence(lag=24), GlobalResidual()], name="persist24"),
        Pipeline(
            [EMOS(), Isotonic(pre_integrate_edges=edges)], name="emos_iso",
        ),
        Pipeline(
            [QuantileReg(n_estimators=200, learning_rate=0.05, random_seed=0)],
            name="qreg",
        ),
    ]
    wf = WalkForward(
        cv="expanding-window", n_folds=5, embargo=24,
        refit_on_full=False,    # demo: only OOF metrics, no retrain.
    )
    # qreg uses the full numeric feature matrix; emos uses the synthetic
    # ensemble columns. The pipeline as-shipped uses ONE X per call, so
    # for the demo we feed X_emos and let qreg model on it too — both
    # trainers happen to handle small-K dense input.
    result = wf.fit_predict(model, X_emos, y, ids=ids, timestamps=ts)

    print("\n=== distribution-level OOF metrics ===")
    print(result.to_table(y, metrics=["crps", "log_score", "pit"]))

    print("\n=== bracket-contract OOF metrics ===")
    print(result.to_table(
        y, metrics=["log_loss_bracket", "brier_bracket"], ladder=ladder,
    ))

    # Skill scores vs each baseline. Two anchors are useful here: the
    # marginal-y "emp" baseline, and the seasonal lag-24 "persist24"
    # baseline. Beating "emp" only means "you learned the marginal";
    # beating "persist24" means "you learned more than the diurnal cycle".
    print("\n=== skill scores (1 - CRPS / CRPS_baseline) ===")
    crps = result.score(y, metrics=["crps"])
    for ref in ("emp", "persist24"):
        base = crps[ref]["crps"]
        print(f"  vs {ref:<10} (CRPS {base:.2f}):")
        for stage, row in crps.items():
            if stage == ref or not np.isfinite(row["crps"]):
                continue
            skill = 1.0 - row["crps"] / base
            print(f"    {stage:<10}  CRPSS = {skill:+.3f}")

    # Pick three OOF rows from the latest fold and show bracket prices.
    print("\n=== example bracket prices for 3 late-fold rows ===")
    bracket_labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi
                      in zip(edges[:-1], edges[1:], strict=True)]
    header = "  ".join(f"{lbl:>7}" for lbl in bracket_labels)
    print(f"{'stage / row':<26}{header}")
    B = edges.shape[0] - 1
    sample_idx = np.array([n - 100, n - 50, n - 10])
    for stage_name, dist in result.items():
        # Map sample_idx into OOF coverage via dist.ids; fall back if absent.
        oof_pos = np.array([np.where(dist.ids == s)[0]
                            for s in sample_idx]).flatten()
        if oof_pos.size != sample_idx.size:
            continue        # row not covered by this stage's OOF
        # Slice the dist for those rows via to-bracket conversion.
        contracts = ladder.price(dist)
        prices = contracts.fair_price.reshape(-1, B)
        for s, p in zip(sample_idx, oof_pos, strict=True):
            cells = "  ".join(f"{v:7.2f}" for v in prices[p])
            print(f"{stage_name} row {s} y={y[s]:>6.0f} {cells}")

    print("\ndone.")


if __name__ == "__main__":
    main()
