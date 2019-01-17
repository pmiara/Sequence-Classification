#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


add_module_names = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = 'Sequence Classification'
copyright = '2019, Jakub Berezowski, Magda Lipowska, Piotr Miara, Grzegorz Szczepaniak'
author = 'Jakub Berezowski, Magda Lipowska, Piotr Miara, Grzegorz Szczepaniak'

version = '1.0'

release = '1.0'

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

todo_include_todos = False

# pip3 install sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
    ]
}

htmlhelp_basename = 'SequenceClassificationdoc'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Sequence Classification.tex', 'Sequence Classification Documentation',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'sequence classification', 'Sequence Classification Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Sequence Classification', 'Sequence Classification Documentation',
     author, 'Sequence Classification', 'System for comparing sequence classifiers',
     'Engineering Thesis'),
]

modindex_common_prefix = ['sequence_classification']

autodoc_default_flags = ['private-members']
