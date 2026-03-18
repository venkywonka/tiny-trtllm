# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import re
import sys

# -- Project information -----------------------------------------------------

project = 'tiny-trtllm'
copyright = f'{datetime.datetime.now().year}, tiny-trtllm contributors'
author = 'tiny-trtllm contributors'
html_show_sphinx = False

# Get the version dynamically from pyproject.toml
_pyproject_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml'))
version = '0.1.0'  # fallback
with open(_pyproject_path) as f:
    _match = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    if _match:
        version = _match.group(1)

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
]

# Prefix section labels with document name to avoid duplicates
autosectionlabel_prefix_document = True

extensions += [
    'sphinx_copybutton',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx_togglebutton',
    'sphinxcontrib.mermaid',
]

autodoc_member_order = 'bysource'
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_model_show_field_list = True
autodoc_pydantic_model_member_order = "groupwise"
autodoc_pydantic_model_hide_pydantic_methods = True
autodoc_pydantic_field_list_validators = False

myst_heading_anchors = 4

myst_enable_extensions = [
    "deflist",
    "substitution",
    "dollarmath",
]

myst_substitutions = {
    "version": version,
}

autosummary_generate = True
copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = ">>> |$ |# "
copybutton_line_continuation_character = "\\"

# -- Options for HTML output -------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'nvidia_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "switcher": {
        "json_url": "./_static/switcher.json",
        "version_match": version,
        "check_switcher": True,
    },
    "extra_footer": [
        f'<p>Last updated on {datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y")}.</p>',
    ],
}

html_css_files = [
    'custom.css',
]
