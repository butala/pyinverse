"""Calibration of the forward Radon operator ``radon_matrix``.

These tests pin down the three things that a caller cannot infer from the
sparse matrix alone, and that the rest of the package (and, downstream, the
Kalman-Wiener filter) depends on:

1. **Row order.**  Rows are blocked detector-major, row ``l*Na + k`` is detector
   sample ``l`` of angle ``k``.  Checked against the independent per-pixel loop
   :meth:`pyinverse.grid.RegularGrid.sinogram`, which stores angle in the
   column and detector in the row.
2. **Normalisation.**  For ``a=0`` a row holds the lengths of intersection of
   the line with the grid cells, so ``R @ x`` is the unit-density line integral
   of the piecewise-constant image ``x``.  Checked against the analytic chord
   profile of a uniform disk and against the mass identity
   (sum over the detector, times the detector spacing, is the image's area).

3. **The finite-beam path** (``a > 0``) is the convolution of the line integral
   with a box of width ``a``.

Everything here runs in-process (``n_cpu=1``), so it needs no ``__main__`` guard
and no compiler, display stack or optional dependency.
"""

from itertools import product

import numpy as np
import pytest
import scipy.signal

from pyinverse import (
    RegularAxis,
    RegularGrid,
    from_image,
    from_sinogram,
    lasserre_available,
    radon_matrix,
    radon_matrix_ij_polytope,
    to_image,
    to_sinogram,
    volume_cal,
)
from pyinverse.rect import srect_2D_proj

DISK_N = 64
DISK_RADIUS = 0.5


def disk_problem(n=DISK_N, n_a=2):
    """A centred unit-density disk on the ``(-1, 1)`` square."""
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n))
    x = grid.centers[0]**2 + grid.centers[1]**2 <= DISK_RADIUS**2
    y = np.zeros(grid_y.shape)
    y[:] = np.where(np.abs(grid_y.axis_y.centers) <= DISK_RADIUS,
                    2*np.sqrt(np.maximum(DISK_RADIUS**2 - grid_y.axis_y.centers**2, 0)),
                    0.0)[:, None]
    return grid, grid_y, np.asarray(x, dtype=float), y


# --------------------------------------------------------------------------
# Row order
# --------------------------------------------------------------------------
def test_row_order_matches_the_per_pixel_sinogram():
    """``R @ x`` has the same (detector, angle) layout as ``grid.sinogram``."""
    n, n_a = 8, 4
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n))
    x = (np.arange(n*n, dtype=float).reshape(n, n) % 7) + 1.0

    R = radon_matrix(grid, grid_y, n_cpu=1)
    reference = grid.sinogram(grid_y, x)          # (Np, Na)
    assert reference.shape == grid_y.shape
    np.testing.assert_allclose(to_sinogram(R @ from_image(x), grid_y), reference,
                               atol=1e-12)


def test_a_delta_at_the_origin_projects_to_the_centre_detector_sample():
    """The image axes are the physical axes: no flip, no transpose."""
    n, n_a = 9, 4                                 # odd n, so a centre pixel exists
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n))
    centre = n // 2
    assert grid.axis_x.centers[centre] == 0.0 and grid.axis_y.centers[centre] == 0.0

    x = np.zeros((n, n))
    x[centre, centre] = 1.0
    R = radon_matrix(grid, grid_y, n_cpu=1)
    sinogram = to_sinogram(R @ from_image(x), grid_y)

    assert np.all(sinogram.argmax(axis=0) == centre)


def test_the_image_axis_is_the_row_of_the_array():
    """A shift along rows (``axis_y``) shows up as ``sin t`` in the sinogram."""
    n, n_a = 9, 4
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n))
    theta = np.radians(grid_y.axis_x.centers)

    for shift in (-2, 2):
        x = np.zeros((n, n))
        x[n // 2 + shift, n // 2] = 1.0
        R = radon_matrix(grid, grid_y, n_cpu=1)
        rows = to_sinogram(R @ from_image(x), grid_y).argmax(axis=0)
        # detector index of the pixel centre, x*cos(theta) + y*sin(theta)
        offset = grid.axis_y.centers[n // 2 + shift]
        expected = np.argmin(np.abs(grid_y.axis_y.centers[:, None]
                                    - offset*np.sin(theta)[None, :]), axis=0)
        assert np.all(np.abs(rows - expected) <= 1)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def test_line_integral_of_a_disk_matches_the_analytic_chord():
    """The line integral of a unit-density disk is ``2 sqrt(r^2 - t^2)``."""
    grid, grid_y, x, chord = disk_problem()
    R = radon_matrix(grid, grid_y, n_cpu=1)
    sinogram = to_sinogram(R @ from_image(x), grid_y)

    for k in range(grid_y.shape[1]):
        residual = np.max(np.abs(sinogram[:, k] - chord[:, k]))
        assert residual < 4e-2, f'angle {k}: residual {residual:.4e}'
    # The two angles of the disk problem are the same profile.
    np.testing.assert_allclose(sinogram[:, 0], sinogram[:, 1], atol=1e-12)


def test_projection_mass_is_the_image_mass():
    """``sum_l p_l * T_det`` is the image's total mass, exactly."""
    grid, grid_y, x, _ = disk_problem()
    R = radon_matrix(grid, grid_y, n_cpu=1)
    sinogram = to_sinogram(R @ from_image(x), grid_y)

    mass = x.sum() * grid.axis_x.T * grid.axis_y.T
    for k in range(grid_y.shape[1]):
        assert abs(sinogram[:, k].sum() * grid_y.axis_y.T - mass) < 1e-12
    # ... and the raster mass is the disk area to the rasterisation floor.
    assert abs(mass - np.pi * DISK_RADIUS**2) < 2e-2


def _assemble(fn, grid, grid_y, **kw):
    """Assemble a sparse operator from a per-pixel ``(data, indices)`` function."""
    import scipy.sparse

    data, indices, indptr = [], [], [0]
    for ij in product(range(grid.shape[0]), range(grid.shape[1])):
        data_ij, indices_ij = fn(grid, grid_y, ij, **kw)
        data.extend(data_ij)
        indices.extend(indices_ij)
        indptr.append(indptr[-1] + len(data_ij))
    n_p, n_a = grid_y.shape
    n_y, n_x = grid.shape
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(n_p*n_a, n_y*n_x))


def test_the_beam_operator_is_the_per_element_bin_average():
    """``a = T_det`` reproduces the geometric polytope intersection exactly.

    The polytope path (:func:`radon_matrix_ij_polytope`) computes each row as
    the area of the intersection of the detector element with the pixel divided
    by the detector spacing -- that is, the line integral *averaged* over the
    element.  It shares no code with the rectangle-function formulas used by
    the analytic path, so agreement to machine precision pins both the scale
    and the support of the beam operator.
    """
    n, n_a, n_p = 6, 2, 24
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-2, 2, n_p))

    R_beam = radon_matrix(grid, grid_y, a=grid_y.axis_y.T, n_cpu=1)
    R_polytope = _assemble(radon_matrix_ij_polytope, grid, grid_y)

    beam = np.asarray(R_beam.todense())
    assert np.count_nonzero(beam) > 0
    np.testing.assert_allclose(beam, np.asarray(R_polytope.todense()), atol=1e-12)


def test_the_beam_operator_tends_to_the_line_integral():
    """The finite beam is a genuine regularisation: ``a -> 0`` is continuous."""
    n, n_a, n_p = 6, 2, 24
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-2, 2, n_p))
    x = ((np.arange(n*n, dtype=float).reshape(n, n) % 5) + 1.0)

    line = radon_matrix(grid, grid_y, a=0, n_cpu=1) @ x.ravel()
    small = radon_matrix(grid, grid_y, a=grid_y.axis_y.T/1000, n_cpu=1) @ x.ravel()
    coarse = radon_matrix(grid, grid_y, a=grid_y.axis_y.T, n_cpu=1) @ x.ravel()

    assert np.linalg.norm(small - line)/np.linalg.norm(line) < 1e-9
    assert np.linalg.norm(coarse - line)/np.linalg.norm(line) > 1e-3


def test_the_finite_beam_is_the_line_integral_convolved_with_a_box():
    """``a > 0`` averages the ``a = 0`` projection over a box of width ``a``.

    Checked against a numerical convolution of :func:`srect_2D_proj`, which
    shares no code with the closed-form branches of
    :func:`pyinverse.rect.rect_conv_radon_rect`.
    """
    n, n_a, n_p, beam = 9, 4, 64, 0.05
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n_p))
    x = np.zeros((n, n))
    x[n // 2, n // 2] = 1.0
    T_x, T_y = grid.axis_x.T, grid.axis_y.T

    R_line = radon_matrix(grid, grid_y, a=0, n_cpu=1)
    R_beam = radon_matrix(grid, grid_y, a=beam, n_cpu=1)
    line = to_sinogram(R_line @ from_image(x), grid_y)
    got = to_sinogram(R_beam @ from_image(x), grid_y)

    # The independent reference: convolve the a=0 profile with the box on a
    # fine t-grid (FFT convolution: the box is as long as the grid), then
    # sample at the detector positions.
    t_fine = np.linspace(-3.0, 3.0, 400001)
    dt = t_fine[1] - t_fine[0]
    box = (np.abs(t_fine) <= beam/2).astype(float)
    for k in range(n_a):
        theta = np.radians(grid_y.axis_x.centers[k])
        profile = srect_2D_proj([theta], t_fine, 1/T_x, 1/T_y)[:, 0]
        smooth = scipy.signal.fftconvolve(profile, box, mode='same') * dt / beam
        expected = np.interp(grid_y.axis_y.centers, t_fine, smooth)
        assert np.max(np.abs(got[:, k] - expected)) < 2e-4

    # The a=0 projection is nonzero on only a few detector samples, the beam
    # widens it by the beam width, and both integrate to the same mass.
    line_nz = np.count_nonzero(np.abs(line[:, 0]) > 1e-12)
    beam_nz = np.count_nonzero(np.abs(got[:, 0]) > 1e-12)
    assert line_nz < n_p // 4
    assert beam_nz > line_nz
    area = T_x * T_y
    for k in range(n_a):
        assert abs(got[:, k].sum() * grid_y.axis_y.T - area) < 1e-3


# --------------------------------------------------------------------------
# The reshaping helpers
# --------------------------------------------------------------------------
def test_vector_helpers_round_trip():
    n, n_a = 8, 3
    grid = RegularGrid.linspace((-1, 1, n), (-1, 1, n))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, n_a, endpoint=False),
                         RegularAxis.linspace(-1, 1, n))

    image = np.arange(n*n, dtype=float).reshape(n, n)
    assert to_image(from_image(image), grid).shape == grid.shape
    np.testing.assert_array_equal(to_image(from_image(image), grid), image)

    sinogram = np.arange(grid_y.shape[0]*grid_y.shape[1], dtype=float)
    sinogram = sinogram.reshape(grid_y.shape)
    np.testing.assert_array_equal(to_sinogram(from_sinogram(sinogram), grid_y), sinogram)


@pytest.mark.parametrize('helper, argument, message', [
    (to_sinogram, np.zeros(5), 'vector must have shape'),
    (from_sinogram, np.zeros(5), 'sinogram must be 2-D'),
    (from_image, np.zeros(5), 'image must be 2-D'),
])
def test_vector_helpers_reject_the_wrong_shape(helper, argument, message):
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, 2, endpoint=False),
                         RegularAxis.linspace(-1, 1, 8))
    RegularGrid.linspace((-1, 1, 8), (-1, 1, 8))
    with pytest.raises(ValueError, match=message):
        if helper is to_sinogram:
            helper(argument, grid_y)
        else:
            helper(argument)


def test_to_image_rejects_the_wrong_length():
    grid = RegularGrid.linspace((-1, 1, 8), (-1, 1, 8))
    with pytest.raises(ValueError, match='vector must have shape'):
        to_image(np.zeros(7), grid)


# --------------------------------------------------------------------------
# Argument handling, and the optional extension
# --------------------------------------------------------------------------
def test_radon_matrix_rejects_a_bad_worker_count():
    grid = RegularGrid.linspace((-1, 1, 4), (-1, 1, 4))
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, 2, endpoint=False),
                         RegularAxis.linspace(-1, 1, 4))
    with pytest.raises(ValueError, match='n_cpu must be a positive integer'):
        radon_matrix(grid, grid_y, n_cpu=0)


def test_volume_cal_rejects_inconsistent_constraints():
    with pytest.raises(ValueError, match=r'A must have shape \(4, 3\)'):
        volume_cal(4, 3, np.eye(3), np.ones(3))
    with pytest.raises(ValueError, match=r'b must have shape \(6,\)'):
        volume_cal(6, 3, np.vstack([np.eye(3), -np.eye(3)]), np.ones(2))


def test_volume_cal_matches_known_volumes():
    square = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    assert np.isclose(volume_cal(4, 2, square, np.array([0, 1, 0, 1], dtype=float)), 1.0)
    triangle = np.array([[-1, 0], [0, -1], [1, 1]], dtype=float)
    assert np.isclose(volume_cal(3, 2, triangle, np.array([0, 0, 1], dtype=float)), 0.5)


def test_the_lasserre_extension_is_optional():
    """``lasserre_available`` answers without raising, whatever the answer is."""
    available = lasserre_available()
    assert isinstance(available, bool)
    if not available:
        from pyinverse import lasserre_vol
        with pytest.raises(ImportError, match='lasserre'):
            lasserre_vol(4, 2, np.eye(2), np.ones(2))


def test_a_missing_lasserre_library_raises_an_actionable_error(monkeypatch, tmp_path):
    """A missing extension is a clear ImportError, not an import-time failure."""
    from pyinverse import volume

    monkeypatch.setattr(volume, '_lasserre_vol_c', None)
    monkeypatch.setattr(volume, 'lasserre_candidates',
                        lambda: iter([tmp_path / 'not_built.so']))
    assert volume.lasserre_available() is False
    with pytest.raises(ImportError, match='PYINVERSE_LASSERRE_DIR'):
        volume.lasserre_vol(4, 2, np.eye(2), np.ones(2))


def test_the_lasserre_library_is_loaded_lazily():
    """Importing the package must not touch the shared library.

    In a fresh interpreter, ``_lasserre_vol_c`` is still ``None`` after
    ``import pyinverse`` whether or not the extension is built -- the whole
    point of making it optional.
    """
    import subprocess
    import sys

    code = ('import pyinverse, pyinverse.volume as v;'
            'assert v._lasserre_vol_c is None, "loaded at import time";'
            'print("lazy")')
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'lazy'


def test_importing_pyinverse_does_not_need_the_heavy_optionals():
    """A minimal, headless, compiler-free install still imports and computes.

    Runs in a subprocess with the optional modules poisoned in ``sys.modules``,
    so this really does exercise the lazy-import boundary rather than whatever
    happens to be importable in the test session.
    """
    import subprocess
    import sys

    blockers = ['vtk', 'imageio', 'matplotlib', 'pyviz3d', 'tqdm']
    code = (
        'import sys\n'
        f'sys.modules.update({{name: None for name in {blockers!r}}})\n'
        'import numpy as np\n'
        'import pyinverse\n'
        'from pyinverse import RegularAxis, RegularGrid, radon_matrix\n'
        'grid = RegularGrid.linspace((-1, 1, 4), (-1, 1, 4))\n'
        'grid_y = RegularGrid(RegularAxis.linspace(0, 180, 2, endpoint=False),\n'
        '                     RegularAxis.linspace(-1, 1, 4))\n'
        'R = radon_matrix(grid, grid_y, n_cpu=1)\n'
        'assert np.asarray(R @ np.ones(16)).shape == (8,)\n'
        'assert pyinverse.volume._lasserre_vol_c is None\n'
        'print("ok", pyinverse.__version__, pyinverse.lasserre_available())\n'
    )
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('ok')
