"""Single source of truth for the package version.

Kept import-free so that ``setuptools`` can read it statically
(see ``[tool.setuptools.dynamic]`` in ``pyproject.toml``).
"""

__version__ = "0.2.0"
