# Configuration file for the Sphinx documentation builder.
#
# PyQuifer - Oscillatory Consciousness Library
# https://github.com/takku1/PyQuifer

import os
import sys

# Add the src directory to the path for autodoc
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'PyQuifer'
copyright = '2026, Mizuki AI Team'
author = 'takku1'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.napoleon',          # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to external docs (PyTorch, NumPy)
    'sphinx.ext.mathjax',           # Render LaTeX math
    'sphinx.ext.githubpages',       # GitHub Pages support
    'sphinx_autodoc_typehints',     # Better type hint rendering
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Templates path
templates_path = ['_templates']

# Patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']
html_title = 'PyQuifer - Oscillatory Consciousness'
html_short_title = 'PyQuifer'
html_logo = None  # Add logo path if available
html_favicon = None

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# -- Extension configuration -------------------------------------------------

# If using sphinx_rtd_theme, you may need to install it:
# pip install sphinx-rtd-theme sphinx-autodoc-typehints

# Mock imports for modules that may not be installed during doc build
autodoc_mock_imports = ['torch', 'numpy', 'noise', 'sklearn', 'scipy']

# Source suffix
source_suffix = '.rst'

# Master doc
master_doc = 'index'

# Pygments style
pygments_style = 'sphinx'
