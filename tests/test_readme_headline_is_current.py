"""The README's headline numbers must match what the example prints.

The README carried HIGH EA +0.4938 for months after the fixture behind it was
known to be wrong. Prose does not fail a test suite, so nothing caught it. This
does: the numbers in the README's headline block are parsed and checked against
the values the example actually computes.

Kept cheap deliberately — it re-runs one example, not the doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"


def _readme_high_rows() -> dict[str, tuple[float, float]]:
    """{label: (brier, ea)} from the README's HIGH block."""
    text = README.read_text()
    block = text[text.index("===== HIGH"):]
    block = block[:block.index("```")]
    rows: dict[str, tuple[float, float]] = {}
    for line in block.splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+)\s+([+-][\d.]+|0\.0000)", line)
        if m:
            rows[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return rows


def test_readme_quotes_some_numbers():
    """Guard the guard: a regex that matches nothing passes vacuously."""
    rows = _readme_high_rows()
    assert len(rows) >= 3, f"parsed only {list(rows)} from the README block"


@pytest.mark.filterwarnings("ignore")
def test_readme_headline_matches_the_example():
    import sys
    sys.path.insert(0, str(ROOT / "bracketlearn" / "examples"))
    from value_vs_accuracy_weather import DATA, _brier, _price  # noqa: E402

    from bracketlearn.score import edge_alignment  # noqa: E402
    from bracketlearn.trainers import EMOS  # noqa: E402

    df = pl.read_parquet(DATA)
    rows = (df.filter(pl.col("side") == "HIGH")
              .sort(["event_date", "station_id"]).to_dicts())
    cut = int(0.6 * len(rows))
    tr, te = rows[:cut], rows[cut:]
    Xtr = np.array([[r["ens_mean"], r["ens_std"]] for r in tr])
    ytr = np.array([r["realized"] for r in tr])
    emos = EMOS(input_form="aggregates", fit_method="crps_nelder_mead").fit(Xtr, ytr)
    X = np.array([[r["ens_mean"], r["ens_std"]] for r in te])
    dist = emos.predict_dist(X, ids=np.arange(len(te)),
                             timestamps=np.arange(len(te), dtype=float))
    q, m, r = _price(dist, te)

    claimed = _readme_high_rows()
    ref = next(k for k in claimed if k.startswith("reference"))
    raw = next(k for k in claimed if k.startswith("EMOS (raw)"))

    assert claimed[ref][0] == pytest.approx(_brier(m, r), abs=5e-5), (
        "README's reference Brier is stale")
    assert claimed[raw][0] == pytest.approx(_brier(q, r), abs=5e-5), (
        "README's EMOS Brier is stale")
    assert claimed[raw][1] == pytest.approx(edge_alignment(q, m, r) * 100,
                                            abs=5e-5), (
        "README's EMOS EA is stale — regenerate the headline block")


def test_no_competing_readme():
    """Two READMEs on main leave a visitor guessing which is current."""
    others = [p.name for p in ROOT.glob("README*.md") if p.name != "README.md"]
    assert not others, f"competing README files: {others}"
