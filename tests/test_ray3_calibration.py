"""Calibration of the 3-D oblique ray transform ``ray3.ray_matrix``.

``radon_matrix`` is pinned by ``test_radon_calibration.py``; this file pins the
*other* forward operator, the oblique parallel-beam path
:func:`pyinverse.ray3.grid_uv2half_planes` -> :func:`pyinverse.ray3.ray_row` ->
:func:`pyinverse.ray3.ray_matrix`, which is what the notebooks' matrix cache is
built from and which ``fbp3_theta0(..., radon_matrices=...)`` consumes.

Three things a caller cannot infer from the sparse matrix alone:

1. **Detector convention.**  For a world point ``p`` (given in ``(x, y, z)``
   order) and ``R = Rotation.from_euler('ZX', [phi, theta])`` -- note the
   order: the operator writes ``from_euler('ZX', [theta, phi])`` and applies it
   to half-space normals stored in ``(z, y, x)`` order, and the two
   transpositions cancel into this -- the detector sample is
   ``u = (R^-1 p)_x`` and ``v = (R^-1 p)_z``, and the beam runs along
   ``R y_hat``.  The half-space pair ``A, b`` is written in ``(z, y, x)`` order
   (see :func:`pyinverse.ray3.regular_axes2polytope`), which is why the
   in-plane detector axes are world ``x`` and ``z``.
2. **Normalisation.**  A row holds ``(voxel cap beam volume) / (pixel area)``,
   so ``H @ x`` is the pixel-averaged path integral: for a unit-density object
   ``sum(H @ x) * du * dv`` is the object's volume, exactly.
3. **The oblique geometry itself** (``theta != 0``): the same identities hold
   for tilted detectors, which is what the ``grid_uv2half_planes`` rotation
   exists for.

Everything here runs in-process (``n_cpu=1``), so it needs no ``__main__``
guard and no optional dependency.  Grids are small: the polytope volume is
exact but not free.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyinverse.angle import Angle
from pyinverse.axes import RegularAxes3
from pyinverse.grid import RegularGrid
from pyinverse.ray3 import grid_uv2half_planes, ray_matrix

#: (theta, phi) pairs, in degrees.  (0, 0) is the trivial geometry; the rest
#: exercise the oblique path, including a detector tilt.
ORIENTATIONS = [(0.0, 0.0), (30.0, 0.0), (0.0, 40.0), (37.0, 63.0)]

BALL_RADIUS = 0.5


def _rot(theta_deg, phi_deg):
    """The rotation matching the operator (see the module docstring): the
    operator's ``from_euler('ZX', [theta, phi])`` acts on ``(z, y, x)``-ordered
    normals, which is this ``from_euler('ZX', [phi, theta])`` on ``(x, y, z)``
    vectors."""
    return Rotation.from_euler('ZX', [phi_deg, theta_deg], degrees=True)


def _det_uv(p_xyz, theta_deg, phi_deg):
    """Detector coordinates of a world point: u = (R^-1 p)_x, v = (R^-1 p)_z."""
    q = _rot(theta_deg, phi_deg).apply(np.asarray(p_xyz, dtype=float), inverse=True)
    return q[0], q[2]


def ball_problem(n=32, n_uv=25, ulim=1.0):
    """A centred unit-density ball in the ``(-1, 1)^3`` cube."""
    axes3 = RegularAxes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, n_uv), (-ulim, ulim, n_uv))
    cz, cy, cx = (ax.centers for ax in
                  (axes3.axis_z, axes3.axis_y, axes3.axis_x))
    r2 = cz[:, None, None]**2 + cy[None, :, None]**2 + cx[None, None, :]**2
    x = (r2 <= BALL_RADIUS**2).astype(float)
    return axes3, grid_uv, x


def _pixel_averaged_chord(grid_uv, r=BALL_RADIUS, n_sub=7):
    """The analytic chord ``2 sqrt(r^2 - u^2 - v^2)`` averaged over each pixel."""
    sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    du, dv = grid_uv.axis_x.T, grid_uv.axis_y.T
    out = np.empty(grid_uv.shape)
    for m in range(grid_uv.shape[0]):
        for n in range(grid_uv.shape[1]):
            us = grid_uv.axis_x.centers[n] + du * sub
            vs = grid_uv.axis_y.centers[m] + dv * sub
            t2 = us[None, :]**2 + vs[:, None]**2
            out[m, n] = (2.0 * np.sqrt(np.maximum(r**2 - t2, 0.0))).mean()
    return out


def _grid_volume(axes3):
    """The volume spanned by the *borders* -- RegularAxis.linspace is a sample
    grid (np.linspace semantics), so the cells reach half a spacing past the
    end points."""
    span = [ax.borders[-1] - ax.borders[0]
            for ax in (axes3.axis_x, axes3.axis_y, axes3.axis_z)]
    return float(np.prod(span))


# --------------------------------------------------------------------------
# Detector convention
# --------------------------------------------------------------------------
def test_a_delta_at_the_origin_projects_to_the_centre_detector_sample():
    """Odd detector: the image axes are the physical axes, no flip/transpose."""
    n, n_uv = 9, 21
    axes3 = RegularAxes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-1, 1, n_uv), (-1, 1, n_uv))
    c3, c_uv = n // 2, n_uv // 2
    for ax in (axes3.axis_x, axes3.axis_y, axes3.axis_z):
        assert ax.centers[c3] == 0.0
    assert grid_uv.axis_x.centers[c_uv] == 0.0
    assert grid_uv.axis_y.centers[c_uv] == 0.0

    x = np.zeros(axes3.shape)
    x[c3, c3, c3] = 1.0
    for theta_deg, phi_deg in ORIENTATIONS:
        H = ray_matrix(Angle(deg=theta_deg), Angle(deg=phi_deg),
                       axes3, grid_uv, n_cpu=1)
        y = (H @ x.ravel()).reshape(grid_uv.shape)
        m, k = np.unravel_index(np.argmax(y), y.shape)
        assert (m, k) == (c_uv, c_uv), (theta_deg, phi_deg, m, k)


@pytest.mark.parametrize('theta_deg,phi_deg', ORIENTATIONS)
def test_a_delta_projects_to_its_detector_coordinates(theta_deg, phi_deg):
    """The peak of a delta sits at u = (R^-1 p)_x, v = (R^-1 p)_z (to a pixel)."""
    n, n_uv = 7, 9
    axes3 = RegularAxes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-1.2, 1.2, n_uv), (-1.2, 1.2, n_uv))
    cz, cy, cx = (ax.centers for ax in
                  (axes3.axis_z, axes3.axis_y, axes3.axis_x))
    H = ray_matrix(Angle(deg=theta_deg), Angle(deg=phi_deg),
                   axes3, grid_uv, n_cpu=1)

    for ijk in [(5, 3, 1), (1, 5, 4), (3, 1, 6), (6, 6, 0)]:
        i, j, k = ijk
        p = np.array([cx[k], cy[j], cz[i]])          # world (x, y, z)
        x = np.zeros(axes3.shape)
        x[ijk] = 1.0
        y = (H @ x.ravel()).reshape(grid_uv.shape)
        m, n_det = np.unravel_index(np.argmax(y), y.shape)

        u, v = _det_uv(p, theta_deg, phi_deg)
        k_u = np.argmin(np.abs(grid_uv.axis_x.centers - u))
        m_v = np.argmin(np.abs(grid_uv.axis_y.centers - v))
        assert abs(n_det - k_u) <= 1, (ijk, theta_deg, phi_deg, n_det, k_u, u)
        assert abs(m - m_v) <= 1, (ijk, theta_deg, phi_deg, m, m_v, v)


def test_the_beam_axis_is_the_inverse_rotated_y_axis():
    """``grid_uv2half_planes``: the prism is unbounded along ``R^-1 y_hat``."""
    n_uv = 5
    grid_uv = RegularGrid.linspace((-1, 1, n_uv), (-1, 1, n_uv))
    for theta_deg, phi_deg in ORIENTATIONS:
        A, b = grid_uv2half_planes(Angle(deg=theta_deg), Angle(deg=phi_deg),
                                   grid_uv, (n_uv // 2, n_uv // 2))
        # the null space of the four half-plane normals is the beam direction;
        # the normals are stored in (z, y, x) order, so reverse to (x, y, z)
        axis = np.linalg.svd(np.asarray(A, dtype=float))[2][-1][::-1]
        expected = _rot(theta_deg, phi_deg).apply([0.0, 1.0, 0.0])
        assert abs(abs(float(axis @ expected)) - 1.0) < 1e-12, (theta_deg, phi_deg)
        assert np.shape(A) == (4, 3) and np.shape(b) == (4,)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize('theta_deg,phi_deg', ORIENTATIONS)
def test_forward_mass_of_a_unit_density_cube_is_its_volume(theta_deg, phi_deg):
    """``sum(H @ 1) * du * dv`` is the grid volume -- exactly, at any angle."""
    n = 6
    axes3 = RegularAxes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    # the detector window must cover the rotated cube's shadow (radius < sqrt 3)
    grid_uv = RegularGrid.linspace((-2.0, 2.0, 11), (-2.0, 2.0, 11))
    x = np.ones(axes3.shape)

    H = ray_matrix(Angle(deg=theta_deg), Angle(deg=phi_deg),
                   axes3, grid_uv, n_cpu=1)
    mass = float((H @ x.ravel()).sum()) * grid_uv.axis_x.T * grid_uv.axis_y.T
    volume = x.sum() * (axes3.axis_x.T * axes3.axis_y.T * axes3.axis_z.T)
    assert abs(mass - volume) < 1e-9 * volume, (theta_deg, phi_deg, mass, volume)
    # ... and that volume is the borders span, not the sample span
    assert abs(volume - _grid_volume(axes3)) < 1e-12 * volume


def test_the_ball_projects_to_the_analytic_chord_profile():
    """``H @ x`` of a unit-density ball is the pixel-averaged chord of a ball."""
    theta_deg, phi_deg = 37.0, 63.0                 # oblique: tilted detector
    axes3, grid_uv, x = ball_problem()
    H = ray_matrix(Angle(deg=theta_deg), Angle(deg=phi_deg),
                   axes3, grid_uv, n_cpu=1)
    y = (H @ x.ravel()).reshape(grid_uv.shape)

    chord = _pixel_averaged_chord(grid_uv)
    # away from the tangential edge the profile is flat enough to compare;
    # the residual left there is the rasterisation of the ball
    core = chord >= 0.6 * 2.0 * BALL_RADIUS
    assert core.any()
    residual = float(np.max(np.abs(y - chord)[core]))
    assert residual < 6e-2, f'residual {residual:.4e}'
    # peak is the diameter
    assert abs(y.max() - 2.0 * BALL_RADIUS) < 5e-2 * 2.0 * BALL_RADIUS
    # ... and the projection mass is the ball volume to the rasterisation floor
    mass = float(y.sum()) * grid_uv.axis_x.T * grid_uv.axis_y.T
    volume = 4.0 / 3.0 * np.pi * BALL_RADIUS**3
    assert abs(mass - volume) < 3e-2 * volume, (mass, volume)


# --------------------------------------------------------------------------
# Worker count contract (mirrors test_radon_matrix_rejects_a_bad_worker_count)
# --------------------------------------------------------------------------
def test_ray_matrix_rejects_a_bad_worker_count():
    axes3 = RegularAxes3.linspace((-1, 1, 4), (-1, 1, 4), (-1, 1, 4))
    grid_uv = RegularGrid.linspace((-2, 2, 3), (-2, 2, 3))
    for n_cpu in (0, -2):
        with pytest.raises(ValueError, match='n_cpu'):
            ray_matrix(Angle(deg=0), Angle(deg=0), axes3, grid_uv, n_cpu=n_cpu)
