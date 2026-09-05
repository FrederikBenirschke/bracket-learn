"""Committed notebooks carry no outputs.

Executed notebooks embed every figure as base64: these four were 2.5 MB, of
which ~2.4 MB was output, and GitHub often refuses to render a file that size.
Figures worth seeing live in ``docs/_static/``; the jupytext pairs under
``notebooks/_src/`` hold the same code as plain .py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


def test_there_are_notebooks_to_check():
    """A glob matching nothing would pass every parametrised test below."""
    assert NOTEBOOKS, "no notebooks found, has the directory moved?"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    offenders = [
        i for i, c in enumerate(nb["cells"])
        if c.get("cell_type") == "code" and c.get("outputs")
    ]
    assert not offenders, (
        f"{nb_path.name} has outputs in cells {offenders}. Strip before "
        f"committing (nbstripout, or clear all outputs in Jupyter)."
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_small(nb_path: Path):
    """A stripped notebook is tens of KB. Anything near a megabyte means
    outputs came back by another route (widget state, attachments)."""
    kb = nb_path.stat().st_size / 1024
    assert kb < 200, f"{nb_path.name} is {kb:.0f} KB; expected < 200 KB stripped"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_still_has_its_code(nb_path: Path):
    """Stripping outputs must not strip source."""
    nb = json.loads(nb_path.read_text())
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert code, f"{nb_path.name} has no code cells left"
    assert any("".join(c.get("source", [])).strip() for c in code), (
        f"{nb_path.name} has code cells but all are empty")


# ---------------------------------------------------------------------------
# jupytext pairing
# ---------------------------------------------------------------------------
#
# The .ipynb and its _src/*.py mirror are a jupytext pair, and nothing kept
# them in step. A repo-wide em-dash removal edited every _src/*.py and left
# the four notebooks untouched, so 36 em-dashes survived in committed
# markdown and in plot TITLES, which are user-visible strings. The tests
# above all passed throughout: they check outputs, size and non-emptiness,
# none of which a stale mirror disturbs.
#
# Comparing rendered output would require executing the notebooks. Comparing
# the prose is enough to catch a one-sided edit, which is the failure that
# actually happened.


def _src_for(nb_path: Path) -> Path:
    return nb_path.parent / "_src" / (nb_path.stem + ".py")


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_a_paired_source(nb_path: Path):
    src = _src_for(nb_path)
    assert src.exists(), (
        f"{nb_path.name} has no jupytext mirror at {src.relative_to(src.parents[2])}. "
        f"Either pair it or drop the notebook."
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_declares_the_pairing(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    fmt = nb.get("metadata", {}).get("jupytext", {}).get("formats", "")
    assert "_src" in fmt, (
        f"{nb_path.name} does not declare its jupytext pairing "
        f"(metadata.jupytext.formats = {fmt!r}), so `jupytext --sync` will "
        f"not keep it in step with its mirror."
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_markdown_matches_its_source(nb_path: Path):
    """Every markdown line in the notebook must appear in the mirror.

    One-directional by design. jupytext writes the mirror's markdown as
    ``# `` comment lines, so the notebook's prose is a subset of the .py
    text; the reverse does not hold, since the .py also carries the yaml
    header and cell delimiters. A line present in the notebook and absent
    from the mirror means the two were edited apart.
    """
    src_text = _src_for(nb_path).read_text()
    nb = json.loads(nb_path.read_text())
    missing = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        for line in "".join(cell.get("source", [])).split("\n"):
            line = line.strip()
            if len(line) > 20 and line not in src_text:
                missing.append(line)
    assert not missing, (
        f"{nb_path.name} has {len(missing)} markdown line(s) absent from its "
        f"_src mirror, so the pair has drifted. Re-sync with\n"
        f"    jupytext --to ipynb notebooks/_src/{nb_path.stem}.py "
        f"-o notebooks/{nb_path.name}\n"
        f"first missing: {missing[0][:100]!r}"
    )
