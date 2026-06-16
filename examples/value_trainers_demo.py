"""Train *for value* and score it: the `bracketlearn.value` trainers + metrics.

End to end on the bundled anonymized weather sample
(`examples/data/weather_value_sample.parquet`):

  1. Build per-row bracket grids and the reference (market) price per bracket.
  2. Fit `BlendedBracketGBM` at several tilts `λ` (objective `L = CE − λ·EA`).
  3. Score each against the reference two ways — `edge_alignment` (value,
     fee-free) and `edge_alignment_costed` (value net of a per-trade fee).
  4. Show the rule: EA rises with the tilt, but *costed* value peaks at an
     interior `λ` — select the tilt by costed value, never by EA.
  5. Same call, torch engine (`BlendedBracketNet`).

Run::

    python -m bracketlearn.examples.value_trainers_demo
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import polars as pl

from bracketlearn.value import (
    BlendedBracketGBM,
    BlendedBracketNet,
    value_report_dist,
)

DATA = os.path.join(os.path.dirname(__file__), "data", "weather_value_sample.parquet")
FEE = 0.0175   # per-contract fee for the costed metric


def load():
    """Rows with a fully-quoted ladder; build the trainer inputs keyed by id."""
    df = pl.read_parquet(DATA).filter(pl.col("side") == "HIGH")
    X, y, ids, brackets_by_id, reference_by_id = [], [], [], {}, {}
    for r in df.to_dicts():
        ref = np.asarray(r["ref_price"], float)
        if not np.all(np.isfinite(ref)):          # skip rows with an unquoted bracket
            continue
        rid = len(ids)
        brackets_by_id[rid] = np.asarray(r["edges"], float)
        reference_by_id[rid] = ref
        X.append([r["ens_mean"], r["ens_std"], r["nws"], r["climo"], r["clim_sigma"]])
        y.append(r["realized"])
        ids.append(rid)
    return (np.array(X), np.array(y), np.array(ids), brackets_by_id, reference_by_id)


def score(dist, reference_by_id, y_by_id):
    """Value (EA) + value-net-of-fee (costed) in ONE call — no manual flatten.

    ``value_report_dist`` does the per-row ragged flatten + renormalization and
    scores against the same ``reference_by_id`` we trained with. We add the model
    Brier (the *accuracy* axis) separately, since it is not part of the value
    report (which reports the reference's Brier, ``A``, instead)."""
    rep = value_report_dist(dist, reference_by_id, y_by_id, fee=FEE)
    briers = []
    for j, rid in enumerate(dist.ids):
        K = len(reference_by_id[rid])
        q = np.nan_to_num(dist.probs[j][:K])
        q = q / q.sum()
        edges = np.asarray(dist.edges[j][: K + 1])
        bi = int(np.clip(np.searchsorted(edges, y_by_id[rid], "right") - 1, 0, K - 1))
        oh = np.zeros(K)
        oh[bi] = 1.0
        briers.append(float(((q - oh) ** 2).sum()))
    return rep["EA"] * 100, rep["costed_mean_pnl"] * 100, float(np.mean(briers))


def fit_score(Trainer, lam, X, y, ids, bbi, rbi, y_by_id, tr, te, **kw):
    # Construction is hyperparameters only; the per-row grids/references flow at
    # call time. fit/predict each select their subset by the ids handed to them —
    # the same dicts drop straight into WalkForward.fit_predict(..., brackets_by_id=).
    model = Trainer(lam=lam, **kw)
    model.fit(X[tr], y[tr], ids=ids[tr], brackets_by_id=bbi, reference_by_id=rbi)
    dist = model.predict_dist(X[te], ids=ids[te], timestamps=ids[te].astype(float),
                              brackets_by_id=bbi)
    return score(dist, rbi, y_by_id)


def main():
    warnings.filterwarnings("ignore")
    X, y, ids, bbi, rbi = load()
    y_by_id = dict(zip(ids.tolist(), y.tolist(), strict=True))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    cut = int(0.6 * len(perm))
    tr, te = perm[:cut], perm[cut:]
    print(f"loaded {len(ids)} fully-quoted HIGH events  (train {len(tr)} / test {len(te)})")
    print("Train L = CE - lam*EA, then score value (EA) and value-net-of-fee (costed).")
    print("EA rises with the tilt; SELECT lam by the costed column, not EA.\n")

    # lighter regularization than the production defaults — this demo has only
    # two raw features (ens mean/std), so the trees need room to respond.
    gbm_kw = dict(min_child_samples=20, reg_lambda=1.0, num_leaves=31, n_estimators=200)
    print(f"  {'model':22s} {'EA×100':>8} {'costed×100':>11} {'Brier':>8}")
    for lam in [0.0, 1.0, 2.0, 4.0, 8.0]:
        ea, costed, brier = fit_score(BlendedBracketGBM, lam, X, y, ids, bbi, rbi, y_by_id,
                                      tr, te, **gbm_kw)
        tag = "  (lam=0: pure-CE baseline)" if lam == 0 else ""
        print(f"  {'gbm  lam=' + str(lam):22s} {ea:>8.3f} {costed:>11.3f} {brier:>8.4f}{tag}")

    print()
    for lam in [0.0, 4.0]:
        ea, costed, brier = fit_score(BlendedBracketNet, lam, X, y, ids, bbi, rbi, y_by_id,
                                      tr, te, epochs=300)
        print(f"  {'net  lam=' + str(lam):22s} {ea:>8.3f} {costed:>11.3f} {brier:>8.4f}")

    print("\nPick lam by the costed column (net of fee), not EA: EA always wants more tilt.")


if __name__ == "__main__":
    main()
