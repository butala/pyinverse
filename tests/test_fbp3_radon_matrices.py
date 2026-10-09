"""Calibration of the ``radon_matrices`` backend of ``fbp3_theta0``.

``fbp3_theta0`` has two ways to backproject the ramp-filtered sinogram:

* without matrices, :func:`pyinverse.fbp3.backproject3` interpolates the
  filtered sinogram at each voxel's projected centre -- a *point sample*;
* with matrices, ``radon_matrices[i].T @ p.flat`` scaled by
  ``alpha = du dv / dV`` is a *voxel average* over the ray matrices of
  :func:`pyinverse.ray3.ray_matrix` (``tests/test_ray3_calibration.py`` pins
  those matrices themselves).

The two are discretisations of the same adjoint and agree to machine precision
on constant fields and to a percent on smooth ones; the ramp filter makes the
filtered sinogram peaky on the scale of a detector pixel, and that is where
they part company (see the tolerances below).  This file pins the wiring of the
matrix path -- the ``alpha`` factor, the transpose as the adjoint, the reshape
onto the reconstruction grid, and the shared ``dphi cos(theta0)`` prefactor --
which is what ``Regularized 3D reconstruction.ipynb`` exercises and which was
previously checked only by running that notebook.

Everything here runs in-process (``n_cpu=1``), so it needs no ``__main__``
guard and no optional dependency.
"""

import numpy as np
import pytest

from pyinverse.angle import Angle, AngleRegularAxis
from pyinverse.axes import RegularAxes3
from pyinverse.fbp3 import backproject3, fbp3_theta0
from pyinverse.grid import RegularGrid
from pyinverse.phantom3 import Phantom3
from pyinverse.ray3 import ray_matrix

#: Small problem: the matrices are the cost, and they are built once.
N, NZ, N_UV, N_PHI = 14, 7, 21, 8


@pytest.fixture(scope='module')
def problem():
    """The grid, the angles and the ray matrices shared by the tests below."""
    axes3 = RegularAxes3.linspace((-1, 1, N), (-1, 1, N), (-1, 1, NZ))
    grid_uv = RegularGrid.linspace((-2, 2, N_UV), (-2, 2, N_UV))
    phi_axis = AngleRegularAxis.linspace(
        Angle(deg=0), Angle(deg=180), N_PHI, endpoint=False)
    theta = Angle(deg=0)                       # the matrices encode theta = 0
    mats = [ray_matrix(theta, phi_i, axes3, grid_uv, n_cpu=1)
            for phi_i in phi_axis]
    return axes3, grid_uv, phi_axis, theta, mats


def _smooth_sinogram(grid_uv, phi_axis, sigma=1.0):
    """A broad Gaussian blob drifting with the angle: peaky only at sub-pixel
    scale, so the two adjoints must agree closely."""
    u = grid_uv.axis_x.centers[None, :]
    v = grid_uv.axis_y.centers[:, None]
    out = []
    for phi_i in phi_axis:
        u0 = 0.3 * np.cos(phi_i.rad)
        v0 = 0.3 * np.sin(phi_i.rad)
        out.append(np.exp(-(((u - u0)**2 + (v - v0)**2) / (2.0 * sigma**2))))
    return out


def _alpha(axes3, grid_uv):
    """The scale of the matrix path: pixel area over voxel volume."""
    return (grid_uv.axis_x.T * grid_uv.axis_y.T
            / (axes3.axis_x.T * axes3.axis_y.T * axes3.axis_z.T))


# --------------------------------------------------------------------------
# The alpha factor and the adjoint
# --------------------------------------------------------------------------
def test_the_matrix_adjoint_maps_a_constant_field_to_one(problem):
    """``alpha * A.T @ 1`` is one in *every* voxel: the scale is exact.

    The beams tile space, so summing the (volume / pixel area) rows over the
    pixels and multiplying by (pixel area / voxel volume) is the fraction of
    the voxel covered -- 1 for a field of ones.  Checked to machine precision.

    This pins the *formula* relating the matrices to a voxel average -- a
    statement about ``ray_matrix``'s rows.  That ``fbp3_theta0`` applies that
    same scale is what ``test_the_two_backends_agree_on_smooth_data`` catches:
    a factor of two inside ``fbp3``'s ``alpha`` passes this test and fails that
    one.
    """
    axes3, grid_uv, phi_axis, theta, mats = problem
    alpha = _alpha(axes3, grid_uv)
    y = np.ones(grid_uv.shape)
    for A in mats:
        M = alpha * (A.T @ y.ravel()).reshape(axes3.shape)
        np.testing.assert_allclose(M, 1.0, rtol=0, atol=1e-12)
        B = backproject3(theta, phi_axis[0], axes3, grid_uv, y)
        np.testing.assert_allclose(M, B, rtol=0, atol=1e-12)


def test_the_two_backends_agree_on_smooth_data(problem):
    """Matrix vs interpolation backprojection through the real API: 0.8 %.

    Pins the wiring of the matrix path (``alpha``, the transpose as the
    adjoint, the reshape) and the prefactor shared with the interpolation
    path.  Measured 8e-3 (sigma = 1.0) and 5e-3 (sigma = 1.6) on this
    problem; the threshold is set with room over that.
    """
    axes3, grid_uv, phi_axis, theta, mats = problem
    sino = _smooth_sinogram(grid_uv, phi_axis)
    X_mat = fbp3_theta0(axes3, grid_uv, phi_axis, sino, radon_matrices=mats,
                        theta0=theta)
    X_int = fbp3_theta0(axes3, grid_uv, phi_axis, sino, theta0=theta)

    assert X_mat.shape == axes3.shape and np.isfinite(X_mat).all()
    rel = np.linalg.norm(X_mat - X_int) / np.linalg.norm(X_int)
    assert rel < 2e-2, f'rel.L2 = {rel:.4e}'
    # ... and there is no residual scale factor between the two
    scale = float((X_mat.ravel() @ X_int.ravel()) / (X_int.ravel() @ X_int.ravel()))
    assert abs(scale - 1.0) < 2e-2, f'scale = {scale:.4f}'


def test_the_two_backends_agree_on_the_phantom_to_the_documented_tolerance(
        problem):
    """The real (peaky) data agree only to ~17 %: the ramp filter's peakiness.

    ``backproject3`` point-samples the filtered sinogram while the matrix path
    voxel-averages it, so they differ wherever the ramp-filtered projection
    varies on the scale of a detector pixel -- measured rel.L2 = 0.17 here
    with best scale 1.016 (a shape difference, not a bias).  This assertion is
    deliberately loose: its job is to catch a transpose, a wrong angle order
    or a factor-of-two, not discretisation.
    """
    axes3, grid_uv, phi_axis, theta, mats = problem
    sino = [Phantom3().proj(theta, Angle(deg=phi_i.deg), grid_uv)
            for phi_i in phi_axis]
    X_mat = fbp3_theta0(axes3, grid_uv, phi_axis, sino, radon_matrices=mats,
                        theta0=theta)
    X_int = fbp3_theta0(axes3, grid_uv, phi_axis, sino, theta0=theta)

    rel = np.linalg.norm(X_mat - X_int) / np.linalg.norm(X_int)
    assert rel < 2.5e-1, f'rel.L2 = {rel:.4e}'
    scale = float((X_mat.ravel() @ X_int.ravel()) / (X_int.ravel() @ X_int.ravel()))
    assert abs(scale - 1.0) < 2.5e-1, f'scale = {scale:.4f}'
    # both must actually reconstruct the object
    assert np.abs(X_mat).max() > 0.1 and np.abs(X_int).max() > 0.1


# --------------------------------------------------------------------------
# The theta0 restriction
# --------------------------------------------------------------------------
def test_a_tilted_detector_is_rejected_with_radon_matrices(problem):
    """The matrices encode the untilted geometry only.

    ``grid_uv2half_planes`` bakes ``theta`` into the half-spaces, so a matrix
    built at theta = 0 silently answers a different question when the detector
    is tilted.  ``fbp3_theta0`` refuses the combination rather than returning
    a wrong-amplitude result (the matrix path carries no ``cos(theta0)``).
    """
    axes3, grid_uv, phi_axis, theta, mats = problem
    sino = _smooth_sinogram(grid_uv, phi_axis)
    with pytest.raises(AssertionError, match='untilted'):
        fbp3_theta0(axes3, grid_uv, phi_axis, sino, radon_matrices=mats,
                    theta0=Angle(deg=15))
    # ... and the same call is fine at theta0 = 0
    fbp3_theta0(axes3, grid_uv, phi_axis, sino, radon_matrices=mats,
                theta0=Angle(deg=0))


def test_the_matrices_are_used_in_angle_order(problem):
    """``radon_matrices[i]`` pairs with ``sinogram3[i]`` and ``phi_axis[i]``.

    Reversing the matrix order must change the result: a silent permutation
    would otherwise look like a reconstruction.
    """
    axes3, grid_uv, phi_axis, theta, mats = problem
    sino = _smooth_sinogram(grid_uv, phi_axis)
    X = fbp3_theta0(axes3, grid_uv, phi_axis, sino, radon_matrices=mats,
                    theta0=theta)
    X_swapped = fbp3_theta0(axes3, grid_uv, phi_axis, sino,
                            radon_matrices=mats[::-1], theta0=theta)
    rel = np.linalg.norm(X_swapped - X) / np.linalg.norm(X)
    assert rel > 1e-2, 'the matrices are not being read in angle order'
