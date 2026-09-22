"""Optional-dependency plumbing.

The numerical core of :mod:`pyinverse` needs only NumPy and SciPy.  Everything
else -- progress bars (``tqdm``), image file I/O (``imageio``), plotting
(``matplotlib``) and 3-D rendering (``vtk`` / ``pyviz3d``) -- is *optional* and
is resolved lazily, at the point of use, so that ``import pyinverse`` and every
analytic operator keep working in a minimal, headless environment.

Use :func:`optional_import` for one-off lazy imports, and the module-level
:func:`tqdm` / :func:`tenumerate` shims for progress reporting (they degrade to
plain iteration when ``tqdm`` is absent).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["TQDM_AVAILABLE", "optional_import", "tenumerate", "tqdm"]


def optional_import(module_name: str, *, extra: str, purpose: str) -> Any:
    """Import *module_name* lazily, with an actionable error if it is absent.

    Args:
        module_name: dotted module to import, e.g. ``'vtk'``.
        extra: name of the ``pyinverse`` optional-dependency group that
            provides it, e.g. ``'viz'``, used in the error message.
        purpose: a short description of what the caller was trying to do,
            used in the error message.

    Returns:
        The imported module.

    Raises:
        ImportError: if *module_name* cannot be imported.  The message names
            the extra to install.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{purpose} requires the optional dependency {module_name!r}, which "
            f"is not installed. Install it with `pip install pyinverse[{extra}]`."
        ) from exc


class _NoOpTqdm:
    """Stand-in for :class:`tqdm.tqdm` used when ``tqdm`` is not installed."""

    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(()) if self.iterable is None else iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def update(self, n=1):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def write(self, s, *args, **kwargs):
        print(s)


def _tenumerate_fallback(iterable, start=0, **kwargs):
    return enumerate(iterable, start=start)


try:  # pragma: no cover - depends on the environment
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - depends on the environment
    _tqdm = _NoOpTqdm

try:  # pragma: no cover - depends on the environment
    from tqdm.contrib import tenumerate as _tenumerate
except ImportError:  # pragma: no cover - depends on the environment
    _tenumerate = _tenumerate_fallback

#: True when the real ``tqdm`` is importable.
TQDM_AVAILABLE = _tqdm is not _NoOpTqdm


def tqdm(iterable=None, **kwargs):
    """``tqdm.tqdm`` when available, otherwise a no-op passthrough."""
    return _tqdm(iterable, **kwargs)


def tenumerate(iterable, **kwargs):
    """``tqdm.contrib.tenumerate`` when available, else :func:`enumerate`."""
    return _tenumerate(iterable, **kwargs)
