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
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'sphinx_copybutton',
    'myst_parser',
]

autodoc_mock_imports = [
    'mmedit.version', 'mmcv.ops.ModulatedDeformConv2d',
    'mmcv.ops.modulated_deform_conv2d', 'mmcv._ext'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

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
                    'description': 'docs at master branch'
                },
                {
                    'name': 'MMEditing 1.x',
                    'url': 'https://mmediting.readthedocs.io/en/dev-1.x/',
                    'description': 'docs at 1.x branch'
                },
            ],
            'active':
            True,
        },
    ],
    'menu_lang':
    'en',
    'header_note': {
        'content':
        'You are reading the documentation for MMEditing 0.x, which '
        'will soon be deprecated by the end of 2022. We recommend you upgrade '
        'to MMEditing 1.0 to enjoy fruitful new features and better performance '  # noqa
        ' brought by OpenMMLab 2.0. Check out the '
        '<a href="https://github.com/open-mmlab/mmediting/releases">changelog</a>, '  # noqa
        '<a href="https://github.com/open-mmlab/mmediting/tree/1.x">code</a> '  # noqa
        'and <a href="https://mmediting.readthedocs.io/en/1.x/">documentation</a> of MMEditing 1.0 for more details.',  # noqa
    }
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
master_doc = 'index'


def builder_inited_handler(app):
    subprocess.run(['bash', './merge_docs.sh'])
    subprocess.run(['python', './stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
