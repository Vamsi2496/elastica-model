# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'elastica_model'
copyright = '2026, authors'
authors = 'A.V. Vamsidhar Reddy & Ramsharan Rangarajan'

extensions = [
    'sphinx.ext.autodoc',        # pull docstrings automatically
    'sphinx.ext.napoleon',       # support NumPy/Google style docstrings
    'sphinx.ext.viewcode',       # add [source] links
]
#html_theme = 'sphinx_rtd_theme'  # Read the Docs theme


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'furo'

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
# Clean CGAL-like styling extras
html_css_files = []
pygments_style = "tango"         # code block style similar to CGAL
pygments_dark_style = "monokai"
html_static_path = ['_static']
autodoc_mock_imports = ["auto", ".f90",".el3"]