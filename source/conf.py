# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the ``lime`` package (repository root) importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))
# Local Sphinx extensions (the interactive demo directive).
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------

project = "LIME Stratified"
author = "Muhammad Rashid, Elvio G. Amparore, Enrico Ferrari, Damiano Verda"
copyright = "2024, Muhammad Rashid and co-authors"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "limestrat_demo",
]

templates_path = []
exclude_patterns = ["demos/*/compute/**"]

autodoc_member_order = "bysource"
autoclass_content = "both"

# The paper uses the same symbols throughout the Method page.
rst_prolog = """
.. |paper| replace:: *Using Stratified Sampling to Improve LIME Image Explanations*
"""

# -- Options for HTML output -------------------------------------------------

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
    html_theme_options = {"navigation_depth": 3}
except ImportError:  # local builds without the theme still work
    html_theme = "alabaster"

html_title = "LIME Stratified"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "rashidrao-pk",
    "github_repo": "lime_stratified",
    "github_version": "master",
    "conf_py_path": "/source/",
}
