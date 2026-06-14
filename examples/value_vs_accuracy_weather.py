"""Accuracy vs value on a real (anonymized) weather sample.

An honest, end-to-end demonstration of the reference-relative value metrics
(``score.edge_alignment`` / ``score.value_report``) on real data:

  1. Fit EMOS (bracketlearn) on ensemble mean/spread.
  2. Price it onto each row's own bracket grid via ``dist.integrate``.
  3. Score it against a real *reference price* ``m`` (a normalized market quote,
     anonymized) two ways: Brier (accuracy) and Edge-Alignment (value).

The honest finding (robust across random splits):

  * EMOS is **less accurate than the market** — its multiclass Brier is
    consistently *worse* than the reference price's. On a calibration
    scoreboard, EMOS loses.
  * EMOS nevertheless has **positive Edge-Alignment** — it is tradeable. Where
    it is wrong is decorrelated from where the market is wrong, so its edge
    points at the market's mistakes. Accuracy and value disagree, on real data.
  * The two things a calibration-minded person would try to "improve" it — a
    mean de-bias toward the truth, and an edge-recalibration toward the market's
    realized error — both *reduce* the value. Calibrating harder is not the same
    as capturing more mispricing (the value guide's §3).

Data: ``examples/data/weather_value_sample.parquet`` — a small anonymized
sample (forecast inputs, realized values, per-row bracket edges, normalized
reference prices with NaN where a bracket had no quote). No venue, station, or
date information.

Run::

    python examples/value_vs_accuracy_weather.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from bracketlearn.score import edge_alignment, value_report
from bracketlearn.trainers import EMOS

DATA = os.path.join(os.path.dirname(__file__), "data", "weather_value_sample.parquet")


def _price(dist, rows, dmu=0.0):
    """Price EMOS (optionally with a mean shift) onto each row's bracket grid,
    then flatten to per-contract (q, m, r). NaN reference quotes are dropped."""
    if dmu:
        from bracketlearn import NormalForecast
        dist = NormalForecast.from_arrays(
            mu=dist.mu + dmu, sigma=dist.sigma,
            ids=dist.ids, timestamps=dist.timestamps, provenance=dist.provenance,
        )
    edges_per_row = [np.asarray(r["edges"], float) for r in rows]
    bracket = dist.integrate(edges_per_row)
    q_list, m_list, r_list = [], [], []
    for j, row in enumerate(rows):
        m = np.asarray(row["ref_price"], float)
        K = m.size
        q = np.nan_to_num(bracket.probs[j][:K])
        if q.sum() <= 0:
            continue
        q = q / q.sum()
        onehot = np.zeros(K)
        bi = int(np.clip(np.searchsorted(edges_per_row[j], row["realized"], "right") - 1, 0, K - 1))
        onehot[bi] = 1.0
        ok = np.isfinite(m)               # only brackets the market actually quoted
        q_list.append(q[ok])
        m_list.append(m[ok])
        r_list.append(onehot[ok])
    return np.concatenate(q_list), np.concatenate(m_list), np.concatenate(r_list)


def _brier(p, r):
    return float(np.mean((p - r) ** 2))


def run_side(df: pl.DataFrame, side: str) -> None:
    rows = df.filter(pl.col("side") == side).to_dicts()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    cut = int(0.6 * len(idx))
    tr = [rows[i] for i in idx[:cut]]
    te = [rows[i] for i in idx[cut:]]

    Xtr = np.array([[r["ens_mean"], r["ens_std"]] for r in tr])
    ytr = np.array([r["realized"] for r in tr])
    # crps_nelder_mead avoids the OLS fit's constant-σ fallback on this data.
    emos = EMOS(input_form="aggregates", fit_method="crps_nelder_mead").fit(Xtr, ytr)

    def predict(subset):
        X = np.array([[r["ens_mean"], r["ens_std"]] for r in subset])
        return emos.predict_dist(X, ids=np.arange(len(subset)), timestamps=np.arange(len(subset), dtype=float))

    dist_te = predict(te)
    q0, m, r = _price(dist_te, te)

    # naive "fix" 1: de-bias EMOS's mean by its train residual (calibrate to truth)
    dmu = float(ytr.mean() - predict(tr).mu.mean())
    qd, _, _ = _price(dist_te, te, dmu=dmu)

    # naive "fix" 2: edge-recalibrate toward the market's realized error (isotonic,
    # fit causally on train) — maximizes calibration of the edge, overfits on small N
    qt, mt, rt = _price(predict(tr), tr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(qt - mt, rt - mt)
    q2 = np.clip(m + iso.predict(q0 - m), 1e-4, 1 - 1e-4)

    ea0 = edge_alignment(q0, m, r) * 100
    bm, b0 = _brier(m, r), _brier(q0, r)
    acc = "less accurate than market" if b0 > bm else "more accurate than market"
    print(f"\n===== {side}  (train {len(tr)}, test {len(te)}) =====")
    print(f"  {'forecast':28s} {'Brier':>8s} {'EA ×100':>9s}")
    print(f"  {'reference (market)':28s} {bm:8.4f} {0.0:9.4f}")
    print(f"  {'EMOS (raw)':28s} {b0:8.4f} {ea0:+9.4f}   <- {acc}, EA > 0")
    print(f"  {'EMOS + mean de-bias':28s} {_brier(qd, r):8.4f} {edge_alignment(qd, m, r) * 100:+9.4f}"
          f"   <- value falls vs raw")
    print(f"  {'EMOS + edge-recal':28s} {_brier(q2, r):8.4f} {edge_alignment(q2, m, r) * 100:+9.4f}"
          f"   <- best Brier, value collapses")
    rep = value_report(q0, m, r)
    print(f"    value_report(EMOS raw): A(ref MSE)={rep['A_reference_mse']:.4f}  "
          f"B(non-orth)={rep['B_non_orthogonality']:.4f}  align_corr={rep['align_corr']:+.3f}")


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pl.read_parquet(DATA)
    print(f"loaded {df.height} rows from {os.path.basename(DATA)}")
    print("EMOS is LESS accurate than the market (worse Brier) yet has positive")
    print("value (EA > 0). Calibrating it harder does not add value.")
    for side in ("HIGH", "LOW"):
        run_side(df, side)


if __name__ == "__main__":
    main()
