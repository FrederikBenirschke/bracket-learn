"""Committed notebooks carry no outputs.

Executed notebooks embed every figure as base64 PNG. The four here were 2.5 MB
of which ~2.4 MB was output: it bloats every clone, makes diffs unreadable
(one re-run rewrites thousands of base64 lines), and GitHub's viewer often
refuses to render a file that large — so the cost was paid and the charts were
not even visible. Figures that should be seen live in ``docs/_static/`` and are
referenced from the docs.

The jupytext pairs under ``notebooks/_src/`` hold the same code as plain .py,
so nothing is lost by stripping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


def test_there_are_notebooks_to_check():
    """Guard the guard: a glob that silently matches nothing passes every
    parametrised test below."""
    assert NOTEBOOKS, "no notebooks found — has the directory moved?"


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
