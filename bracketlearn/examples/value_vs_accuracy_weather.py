"""Accuracy vs value: EMOS against a real market reference.

End-to-end demonstration of the reference-relative value metrics
(``score.edge_alignment`` / ``score.value_report``):

  1. Fit EMOS on multi-model ensemble mean/spread.
  2. Price it onto each row's own bracket grid via ``dist.integrate``.
  3. Score it against the market's normalized mid ``m`` two ways: Brier
     (accuracy) and Edge-Alignment (value).

What this sample shows: EMOS is less accurate than the market (worse Brier)
AND negative-EA, with a 95% interval that crosses zero. The synthetic case in
the value guide's §5, worse Brier, still positive value, does not reproduce
here. What does hold is that the two axes move independently: of the two
calibration "fixes" below, one improves Brier while raising EA and the other
barely moves Brier while lowering it. That is the mechanism the metric exists
to expose, and it does not depend on the sign.

An earlier version of this file claimed positive EA that was "robust across
random splits". That rested on a fixture whose bracket edges were corrupted,
and on a random split of an autocorrelated series. Both are fixed; the claim is
withdrawn. See ``docs/guides/value_vs_accuracy.md`` §5b for the decomposition.

Data: ``examples/data/weather_value_sample.parquet``, 5,429 station-days of
Kalshi weather contracts over 2026-03-17..09-03 across 18 stations, carrying
forecast inputs, realized values, per-row bracket edges (open tails as ±inf),
and normalized reference prices with NaN where a bracket had no quote. Built by
a committed exporter; see the ``.provenance.json`` sidecar for the source
commit. The reference price is a bid-ask midpoint, frictionless, necessary for
value, not sufficient for profit.

Run::

    python examples/value_vs_accuracy_weather.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from bracketlearn.score import bootstrap_ci, edge_alignment, value_report
from bracketlearn.trainers import EMOS

DATA = os.path.join(os.path.dirname(__file__), "data", "weather_value_sample.parquet")


def _clusters(rows):
    """One label per CONTRACT, naming its (station, day) ladder, the unit the
    outcome is shared over. Mirrors _price's flattening exactly, including the
    finite-quote mask, or the labels would not line up with the contracts."""
    out = []
    for row in rows:
        m = np.asarray(row["ref_price"], float)
        n_ok = int(np.isfinite(m).sum())
        out.append(np.full(n_ok, f"{row['station_id']}|{row['event_date']}"))
    return np.concatenate(out) if out else np.array([])


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
    # Chronological split. Weather is strongly autocorrelated day to day, so a
    # random permutation puts adjacent days on opposite sides of the boundary
    # and leaks. The earlier version of this example permuted; the fixture now
    # carries event_date, so it does not have to.
    rows = (df.filter(pl.col("side") == side)
              .sort(["event_date", "station_id"]).to_dicts())
    cut = int(0.6 * len(rows))
    tr, te = rows[:cut], rows[cut:]

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

    # Contracts on one ladder share a single realized temperature, so the CI
    # is clustered by station-day rather than by contract.
    ci = bootstrap_ci(edge_alignment, q0, m, r, cluster=_clusters(te),
                      n_boot=2000, seed=0)
    ea0 = edge_alignment(q0, m, r) * 100
    bm, b0 = _brier(m, r), _brier(q0, r)
    acc = "less accurate than market" if b0 > bm else "more accurate than market"
    print(f"\n===== {side}  (train {len(tr)}, test {len(te)}) =====")
    print(f"  {'forecast':28s} {'Brier':>8s} {'EA ×100':>9s}")
    print(f"  {'reference (market)':28s} {bm:8.4f} {0.0:9.4f}")
    ead, ea2 = edge_alignment(qd, m, r) * 100, edge_alignment(q2, m, r) * 100
    bd, b2 = _brier(qd, r), _brier(q2, r)
    print(f"  {'EMOS (raw)':28s} {b0:8.4f} {ea0:+9.4f}   <- {acc}, "
          f"EA {'>' if ea0 > 0 else '<'} 0")
    print(f"  {'EMOS + mean de-bias':28s} {bd:8.4f} {ead:+9.4f}"
          f"   <- Brier {'falls' if bd < b0 else 'rises'}, "
          f"EA {'falls' if ead < ea0 else 'rises'} vs raw")
    print(f"  {'EMOS + edge-recal':28s} {b2:8.4f} {ea2:+9.4f}"
          f"   <- Brier {'falls' if b2 < b0 else 'rises'}, "
          f"EA {'falls' if ea2 < ea0 else 'rises'} vs raw")
    print(f"    EA 95% CI (clustered by station-day, {ci['n_clusters']:.0f} "
          f"clusters / {ci['n_obs']:.0f} contracts): "
          f"[{ci['lo'] * 100:+.4f}, {ci['hi'] * 100:+.4f}]"
          f"{'  (crosses zero)' if ci['lo'] < 0 < ci['hi'] else ''}")
    rep = value_report(q0, m, r)
    print(f"    value_report(EMOS raw): A(ref MSE)={rep['A_reference_mse']:.4f}  "
          f"B(non-orth)={rep['B_non_orthogonality']:.4f}  align_corr={rep['align_corr']:+.3f}")


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pl.read_parquet(DATA)
    print(f"loaded {df.height} rows from {os.path.basename(DATA)}")
    print(f"window {df['event_date'].min()} .. {df['event_date'].max()}, "
          f"{df['station_id'].n_unique()} stations, chronological 60/40 split")
    print("Brier measures ACCURACY (distance to truth); EA measures VALUE "
          "(whether\nthe deviations from the market price point the right "
          "way). They are\ndifferent quantities and can rank the same "
          "forecasts oppositely.")
    for side in ("HIGH", "LOW"):
        run_side(df, side)


if __name__ == "__main__":
    main()
