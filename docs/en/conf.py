# Copyright (c) OpenMMLab. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MMEditing'
copyright = '2020, MMEditing Authors'
author = 'MMEditing Authors'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc.typehints',
    'sphinx_tabs.tabs',
    'notfound.extension',
]

autodoc_mock_imports = [
    'mmedit.version',
    'mmcv._ext',
    'mmcv.ops.ModulatedDeformConv2d',
    'mmcv.ops.modulated_deform_conv2d',
]

autodoc_skip_member = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = 'both'  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
#autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmediting',
        },
        {
            'name':
            'Version',
            'children': [
                {
                    'name': 'MMEditing 0.x',
                    'url': 'https://mmediting.readthedocs.io/en/latest/',
                    'description': 'Main branch'
                },
                {
                    'name': 'MMEditing 1.x',
                    'url': 'https://mmediting.readthedocs.io/en/1.x/',
                    'description': '1.x branch',
                },
            ],
            'active':
            True,
        },
    ],
    'menu_lang':
    'en',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 3

language = 'en'

# The master toctree document.
root_doc = 'index'
notfound_template = '404.html'


def builder_inited_handler(app):
    subprocess.run(['bash', './.dev_scripts/update_dataset_zoo.sh'])
    subprocess.run(['python', './.dev_scripts/update_model_zoo.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
