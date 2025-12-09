import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "practice"
author = "autumn63"

extensions = [
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

