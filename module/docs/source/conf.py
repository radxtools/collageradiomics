"""Build documentation from the checked-out package and its single version file."""

import sys
from pathlib import Path

module_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(module_dir))
project = "collageradiomics"
author = "BrIC Laboratory"
copyright = "2020, BrIC Laboratory"
release = (module_dir / "VERSION").read_text(encoding="utf-8").strip()
version = release
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
html_theme = "sphinx_rtd_theme"
html_static_path = []
master_doc = "index"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
