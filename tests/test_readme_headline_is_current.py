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
    """Compare against what the EXAMPLE prints, not a reimplementation.

    An earlier version of this test re-derived the split and fit here, which
    pinned the README to a copy of the example rather than to the example. A
    change to run_side's split fraction would have left this passing.
    """
    import io
    import runpy
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        runpy.run_module("bracketlearn.examples.value_vs_accuracy_weather",
                         run_name="__main__")
    out = buf.getvalue()
    start = out.index("===== HIGH")
    nxt = out.find("===== LOW", start)
    high = out[start:nxt if nxt != -1 else len(out)]

    printed: dict[str, tuple[float, float]] = {}
    for line in high.splitlines():
        m = re.match(r"\s{2}(\S.*?)\s{2,}([\d.]+)\s+([+-][\d.]+|0\.0000)", line)
        if m:
            printed[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    assert len(printed) >= 3, f"parsed only {list(printed)} from example output"

    claimed = _readme_high_rows()
    for label, (brier, ea) in claimed.items():
        assert label in printed, (
            f"README row {label!r} is not in the example's output")
        assert brier == pytest.approx(printed[label][0], abs=5e-5), (
            f"README Brier for {label!r} is stale")
        assert ea == pytest.approx(printed[label][1], abs=5e-5), (
            f"README EA for {label!r} is stale — regenerate the headline block")


def test_no_competing_readme():
    """Two READMEs on main leave a visitor guessing which is current."""
    others = [p.name for p in ROOT.glob("README*.md") if p.name != "README.md"]
    assert not others, f"competing README files: {others}"
