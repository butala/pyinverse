"""pyinverse: a lightweight inverse-problems and tomography testbed.

The numerical core -- coordinate axes, grids, analytic phantoms and the
Radon / ray transforms and their backprojection -- depends on **NumPy and
SciPy only**.  Everything else is optional and resolved lazily, at the point of
use:

============  ==========================================  =====================
extra         provides                                    used for
============  ==========================================  =====================
``view``      ``matplotlib``                              ``Grid.plot``/``imshow``
``image``     ``imageio``                                 ``RegularGrid.from_image``
``viz``       ``vtk``, ``pyviz4d``       ``RegularAxes3.actor``/``volume``/``isosurface_actor``
``progress``  ``tqdm``                                    progress bars (optional)
``test``      ``pytest``                                  the test suite
============  ==========================================  =====================

The optional ``lasserre`` C extension (see :func:`pyinverse.volume.lasserre_vol`)
is not needed by any analytic path; it is only an alternative implementation of
the polytope-volume primitive.

Contents
--------
Geometry
    :class:`RegularAxis`, :class:`Angle`, :class:`AngleRegularAxis`,
    :class:`Frequency`, :class:`FrequencyRegularAxis`, :class:`RegularGrid`,
    :class:`RegularAxes3`
Forward operators
    :func:`radon_matrix`, :func:`ray_matrix`, :func:`RegularGrid.sinogram`
Reconstruction
    :func:`fbp`, :func:`fbp3_theta0`, :func:`backproject3`, :func:`ramp_filter`,
    :func:`ramp_filter3`
Phantoms
    :class:`Ellipse`, :class:`Ellipsoid`, :class:`Phantom`, :class:`Phantom3`
Vector reshaping
    :func:`to_sinogram`, :func:`from_sinogram`, :func:`to_image`, :func:`from_image`
"""

import importlib

from ._version import __version__

# Public names resolve lazily (PEP 562).  Beyond import time there is a second
# reason: `import pyinverse` must not leave these modules in ``sys.modules``
# before ``python -m pyinverse.phantom3`` (and the other demos) execute them as
# ``__main__`` -- runpy warns about exactly that and calls the behaviour
# unpredictable.
_EXPORTS = {
    "Angle": ".angle",
    "AngleRegularAxis": ".angle",
    "BackProjector": ".fbp",
    "FFTRegularAxis": ".axis",
    "Frequency": ".frequency",
    "FrequencyRegularAxis": ".frequency",
    "Order": ".axis",
    "Phantom": ".phantom",
    "Phantom3": ".phantom3",
    "RFFTRegularAxis": ".axis",
    "RegularAxes3": ".axes",
    "RegularAxis": ".axis",
    "RegularGrid": ".grid",
    "Ellipse": ".ellipse",
    "Ellipsoid": ".ellipsoid",
    "backproject3": ".fbp3",
    "besinc": ".util",
    "fbp": ".fbp",
    "fbp3_theta0": ".fbp3",
    "from_image": ".radon",
    "from_sinogram": ".radon",
    "lasserre_available": ".volume",
    "lasserre_vol": ".volume",
    "radon_matrix": ".radon",
    "radon_matrix_ij_analytic": ".radon",
    "radon_matrix_ij_polytope": ".radon",
    "ramp_filter": ".fbp",
    "ramp_filter3": ".fbp3",
    "ray_matrix": ".ray3",
    "to_image": ".radon",
    "to_sinogram": ".radon",
    "volume_cal": ".volume",
}


def __getattr__(name):
    """Import the submodule that provides *name*, on first access (PEP 562)."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module, __name__), name)


def __dir__():
    return sorted({*globals(), *_EXPORTS})

__all__ = [
    "Angle",
    "AngleRegularAxis",
    "BackProjector",
    "Ellipse",
    "Ellipsoid",
    "FFTRegularAxis",
    "Frequency",
    "FrequencyRegularAxis",
    "Order",
    "Phantom",
    "Phantom3",
    "RFFTRegularAxis",
    "RegularAxes3",
    "RegularAxis",
    "RegularGrid",
    "__version__",
    "backproject3",
    "besinc",
    "fbp",
    "fbp3_theta0",
    "from_image",
    "from_sinogram",
    "lasserre_available",
    "lasserre_vol",
    "radon_matrix",
    "radon_matrix_ij_analytic",
    "radon_matrix_ij_polytope",
    "ramp_filter",
    "ramp_filter3",
    "ray_matrix",
    "to_image",
    "to_sinogram",
    "volume_cal",
]
