"""Setuptools entry point for the optional ``lasserre`` C extension.

All package metadata lives in ``pyproject.toml``; this file exists only to
declare the extension.  The package is fully functional without it -- only
:func:`pyinverse.volume.lasserre_vol`, an alternative polytope-volume
primitive, uses it, and the shared library is loaded lazily at call time -- so
a missing compiler must never break ``pip install``.  :class:`OptionalBuildExt`
downgrades a failed compilation to a warning.
"""

import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build the extension when possible, but never fail the whole install."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            warnings.warn(
                f"not building the optional 'lasserre' C extension ({exc}). "
                "The analytic Radon transform does not need it; "
                "pyinverse.volume.lasserre_vol will raise ImportError."
            )

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            warnings.warn(
                f"not building the optional extension {ext.name!r} ({exc})."
            )


setup(
    ext_modules=[
        Extension(
            name="lasserre",
            sources=["pyinverse/lasserre/lasserre.c", "pyinverse/lasserre/util.c"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
