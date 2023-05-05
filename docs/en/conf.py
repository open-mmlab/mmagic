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

project = 'MMagic'
copyright = '2023, MMagic Authors'
author = 'MMagic Authors'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_markdown_tables',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'myst_parser',
]

extensions.append('notfound.extension')  # enable customizing not-found page

extensions.append('autoapi.extension')
autoapi_type = 'python'
autoapi_dirs = ['../../mmagic']
autoapi_add_toctree_entry = False
autoapi_template_dir = '_templates'
# autoapi_options = ['members', 'undoc-members', 'show-module-summary']

# # Core library for html generation from docstrings
# extensions.append('sphinx.ext.autodoc')
# extensions.append('sphinx.ext.autodoc.typehints')
# # Enable 'expensive' imports for sphinx_autodoc_typehints
# set_type_checking_flag = True
# # Sphinx-native method. Not as good as sphinx_autodoc_typehints
# autodoc_typehints = "description"

# extensions.append('sphinx.ext.autosummary') # Create neat summary tables
# autosummary_generate = True  # Turn on sphinx.ext.autosummary
# # Add __init__ doc (ie. params) to class summaries
# autoclass_content = 'both'
# autodoc_skip_member = []
# # If no docstring, inherit from base class
# autodoc_inherit_docstrings = True

autodoc_mock_imports = [
    'mmagic.version', 'mmcv._ext', 'mmcv.ops.ModulatedDeformConv2d',
    'mmcv.ops.modulated_deform_conv2d', 'clip', 'resize_right', 'pandas'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# # Remove 'view source code' from top of page (for html, not python)
# html_show_sourcelink = False
# nbsphinx_allow_errors = True  # Continue through Jupyter errors
# add_module_names = False  # Remove namespaces from class/method signatures

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
            'url': 'https://github.com/open-mmlab/mmagic',
        },
        {
            'name':
            'Version',
            'children': [
                {
                    'name': 'MMagic 1.x',
                    'url': 'https://mmagic.readthedocs.io/en/latest/',
                    'description': 'Main branch'
                },
                {
                    'name': 'MMEditing 0.x',
                    'url': 'https://mmagic.readthedocs.io/en/0.x/',
                    'description': '0.x branch',
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
    subprocess.run(['python', './.dev_scripts/update_model_zoo.py'])
    subprocess.run(['python', './.dev_scripts/update_dataset_zoo.py'])


def skip_member(app, what, name, obj, skip, options):
    if what == 'package' or what == 'module':
        skip = True
    return skip


def viewcode_follow_imported(app, modname, attribute):
    fullname = f'{modname}.{attribute}'
    all_objects = app.env.autoapi_all_objects
    if fullname not in all_objects:
        return None

    if all_objects[fullname].obj.get('type') == 'method':
        fullname = fullname[:fullname.rfind('.')]
        attribute = attribute[:attribute.rfind('.')]
    while all_objects[fullname].obj.get('original_path', '') != '':
        fullname = all_objects[fullname].obj.get('original_path')

    orig_path = fullname
    if orig_path.endswith(attribute):
        return orig_path[:-len(attribute) - 1]

    return modname


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
    app.connect('autoapi-skip-member', skip_member)
    if 'viewcode-follow-imported' in app.events.events:
        app.connect(
            'viewcode-follow-imported', viewcode_follow_imported, priority=0)
