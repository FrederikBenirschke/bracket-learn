"""Sphinx configuration for bracketlearn."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

project = "bracketlearn"
author = "Frederik Benirschke"
copyright = f"{datetime.now().year}, {author}"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} v{release}"
html_theme_options = {
    "source_repository": "https://github.com/FrederikBenirschke/bracketlearn",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- autodoc ----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# -- intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- myst ------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
