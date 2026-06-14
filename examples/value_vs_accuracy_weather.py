"""Accuracy vs value on a real (anonymized) weather sample.

An honest, end-to-end demonstration of the reference-relative value metrics
(``score.edge_alignment`` / ``score.value_report``) on real data:

  1. Fit EMOS (bracketlearn) on ensemble mean/spread.
  2. Price it onto each row's own bracket grid via ``dist.integrate``.
  3. Score it two ways against a real *reference price* ``m`` (a normalized
     market quote, anonymized): Brier (accuracy) and Edge-Alignment (value).

The honest finding:

  * Raw EMOS is *less accurate* than the reference (higher Brier) yet still
    carries positive tradeable value (EA > 0) — accuracy and value are
    different axes.
  * Raw EMOS is **over-dispersed** (its σ is too wide), so it leaves value on
    the table. A market-blind, proper-score (CRPS) σ-recalibration fit on the
    training split — pure calibration, it never looks at the reference price —
    sharpens the forecast and roughly doubles the test-set value. Calibrating
    the *dispersion* is the sizing lever of the value guide's §8.

Data: ``examples/data/weather_value_sample.parquet`` — a small anonymized
sample (forecast inputs, realized values, per-row bracket edges, normalized
reference prices). No venue, station, or date information.

Run::

    python examples/value_vs_accuracy_weather.py
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

from bracketlearn import NormalForecast
from bracketlearn.score import crps_gaussian, edge_alignment, value_report
from bracketlearn.trainers import EMOS

DATA = os.path.join(os.path.dirname(__file__), "data", "weather_value_sample.parquet")


def _flatten(dist, rows, scale: float = 1.0):
    """Price ``dist`` (optionally σ-scaled) onto each row's bracket grid and
    flatten to per-contract (q, m, r) arrays paired with the reference price."""
    if scale != 1.0:
        dist = NormalForecast.from_arrays(
            mu=dist.mu, sigma=dist.sigma * scale,
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
        onehot[int(np.clip(np.searchsorted(edges_per_row[j], row["realized"], "right") - 1, 0, K - 1))] = 1.0
        q_list.append(q)
        m_list.append(m)
        r_list.append(onehot)
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

    def design(subset):
        X = np.array([[r["ens_mean"], r["ens_std"]] for r in subset])
        y = np.array([r["realized"] for r in subset])
        return X, y

    Xtr, ytr = design(tr)
    Xte, yte = design(te)
    emos = EMOS(input_form="aggregates").fit(Xtr, ytr)

    dist_tr = emos.predict_dist(Xtr, ids=np.arange(len(tr)), timestamps=np.arange(len(tr), dtype=float))
    dist_te = emos.predict_dist(Xte, ids=np.arange(len(te)), timestamps=np.arange(len(te), dtype=float))

    # fit the σ-scale on TRAIN by CRPS (a proper score — never sees the reference)
    scales = np.linspace(0.4, 1.4, 51)
    crps_tr = [float(crps_gaussian(
        NormalForecast.from_arrays(mu=dist_tr.mu, sigma=dist_tr.sigma * s,
                                   ids=dist_tr.ids, timestamps=dist_tr.timestamps,
                                   provenance=dist_tr.provenance), ytr).mean())
        for s in scales]
    best = float(scales[int(np.argmin(crps_tr))])

    q0, m, r = _flatten(dist_te, te, scale=1.0)
    qb, _, _ = _flatten(dist_te, te, scale=best)

    print(f"\n===== {side}  (train {len(tr)}, test {len(te)}) =====")
    print(f"  reference (market)  Brier {_brier(m, r):.4f}   EA  0.0000")
    print(f"  raw EMOS            Brier {_brier(q0, r):.4f}   EA {edge_alignment(q0, m, r) * 100:+.4f}  (×100)")
    print(f"  σ-recal (CRPS, ×{best:.2f}) Brier {_brier(qb, r):.4f}   EA {edge_alignment(qb, m, r) * 100:+.4f}  (×100)")
    rep = value_report(qb, m, r)
    print(f"    value_report(σ-recal): A(ref MSE)={rep['A_reference_mse']:.4f}  "
          f"B(non-orth)={rep['B_non_orthogonality']:.4f}  align_corr={rep['align_corr']:+.3f}")


def main() -> None:
    df = pl.read_parquet(DATA)
    print(f"loaded {df.height} rows from {os.path.basename(DATA)}")
    print("Raw EMOS is over-dispersed: less accurate than the market, yet it has")
    print("tradeable value. A market-blind σ-recalibration recovers more of it.")
    for side in ("HIGH", "LOW"):
        run_side(df, side)


if __name__ == "__main__":
    main()
