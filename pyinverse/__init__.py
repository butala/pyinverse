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
``viz``       ``vtk``, ``pyviz3d``                        ``RegularAxes3.actor``/``volume``
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

from ._version import __version__
from .angle import Angle, AngleRegularAxis
from .axes import RegularAxes3
from .axis import FFTRegularAxis, Order, RegularAxis, RFFTRegularAxis
from .ellipse import Ellipse
from .ellipsoid import Ellipsoid
from .fbp import BackProjector, fbp, ramp_filter
from .fbp3 import backproject3, fbp3_theta0, ramp_filter3
from .frequency import Frequency, FrequencyRegularAxis
from .grid import RegularGrid
from .phantom import Phantom
from .phantom3 import Phantom3
from .radon import (
    from_image,
    from_sinogram,
    radon_matrix,
    radon_matrix_ij_analytic,
    radon_matrix_ij_polytope,
    to_image,
    to_sinogram,
)
from .ray3 import ray_matrix
from .util import besinc
from .volume import lasserre_available, lasserre_vol, volume_cal

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
