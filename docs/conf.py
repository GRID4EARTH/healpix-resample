# -- Project information -----------------------------------------------------
import datetime as dt
import pathlib
import subprocess

project = "healpix-resample"
author = f"{project} developers"
initial_year = "2026"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

# The root toctree document.
root_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_nb",
]

extlinks = {
    "issue": ("https://github.com/GRID4EARTH/healpix-resample/issues/%s", "GH%s"),
    "pull": ("https://github.com/GRID4EARTH/healpix-resample/pull/%s", "PR%s"),
}

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

nb_execution_mode = "auto"

# Cache execution results to speed up rebuilds
nb_execution_cache_path = "_build/.jupyter_cache"

# Raise error if notebook execution fails
nb_execution_raise_on_error = True

# Execution timeout in seconds
nb_execution_timeout = 120

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "directory", ".ipynb_checkpoints", "jupyter_execute"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_theme_options = {
    "github_url": "https://github.com/GRID4EARTH/healpix-resample",
    "icon_links_label": "Quick Links",
}

# -- Options for the intersphinx extension -----------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autosummary_generate = True
autodoc_typehints = "none"

# Suppress warnings for inherited methods in autosummary-generated pages
nitpick_ignore_regex = [
    (r"py:(obj|meth)", r"healpix_resample\.(\w+\.)*\w+\.(comp_matrix|get_cell_ids|invert|resample)"),
]

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
