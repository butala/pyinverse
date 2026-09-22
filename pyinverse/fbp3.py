import numpy as np
import scipy as sp

from ._optional import tenumerate
from .angle import Angle


def ramp_filter3(grid_uv_ft_Hz):
    """
    Ramp filter for 3-D FBP with a planar 2-D detector rotating about z.

    Writing the object's Fourier transform in the detector frame,
    f = f_u e1 + f_v e2, the change of variables (phi, f_u, f_v) has
    Jacobian |f_u|, so the reconstruction filter is the *1-D* ramp
    |f_u| applied along the detector axis transverse to the rotation
    axis.  It is *flat* along f_v: the detector rows v = z are
    independent 2-D Radon data sets for the slices f(., ., z), so no
    weighting may be applied across rows.

    In particular this is *not* the isotropic 2-D ramp
    sqrt(f_u**2 + f_v**2), which would impose a frequency dependent
    gain sqrt(fx**2 + fy**2 + fz**2) / sqrt(fx**2 + fy**2) != 1 and
    hence a reconstruction bias that does not decrease with the number
    of projections.

    See doc/fbp3-filter-derivation.org for the derivation.
    """
    Cu_Hz = grid_uv_ft_Hz.centers[0]
    return np.abs(Cu_Hz)


def backproject3(theta, phi, axes3, grid_uv, X, method='linear'):
    """
    """
    c_z, c_y, c_x = axes3.centers

    c_phi = phi.cos
    s_phi = phi.sin

    c_theta = theta.cos
    s_theta = theta.sin

    e1 = np.array([c_phi, s_phi, 0])
    e2 = np.array([s_phi * s_theta, -c_phi * s_theta, c_theta])

    p_xyz = np.array([c_x, c_y, c_z])
    e12 = np.array([e1, e2])

    uv_backproject = np.einsum('ij,jklm->iklm', e12, p_xyz)

    u_backproject = uv_backproject[0, :, :, :]
    v_backproject = uv_backproject[1, :, :, :]

    interp2d = sp.interpolate.RegularGridInterpolator(
        (grid_uv.axis_y.centers, grid_uv.axis_x.centers), X,
        method=method, bounds_error=True)

    X_backproject = interp2d(np.array([v_backproject.flatten(), u_backproject.flatten()]).T)
    X_backproject.shape = axes3.shape

    return X_backproject


#: Default detector tilt; a module-level singleton so that the listener-free
#: default argument cannot be mutated by a caller.
_THETA0_UNTILTED = Angle(deg=0)


def fbp3_theta0(axes3, grid_uv, phi_axis, sinogram3, radon_matrices=None,
                theta0=_THETA0_UNTILTED):
    """
    Filtered backprojection for a planar 2-D detector rotating about z.

    Parameters
    ----------
    axes3 : RegularAxes3-like
        Reconstruction grid; ``axes3.shape == (Nz, Ny, Nx)``.
    grid_uv : RegularGrid
        Detector grid: ``axis_x`` -> u (transverse to the rotation axis),
        ``axis_y`` -> v.
    phi_axis : AngleRegularAxis
        Rotation angles.
    sinogram3 : sequence of ndarray
        ``sinogram3[i]`` is the projection p(u, v) measured at
        ``phi_axis[i]``; each has shape ``grid_uv.shape``.
    radon_matrices : sequence, optional
        Pre-computed backprojection matrices.  These encode the *untilted*
        (theta = 0) geometry only -- they carry no theta dependence at
        all -- so they may be combined only with ``theta0 = 0``.
    theta0 : Angle, optional
        Fixed polar tilt of the detector relative to the rotation (z)
        axis.

    Notes
    -----
    The result carries an explicit ``cos(theta0)`` factor.  Writing the
    object's Fourier transform in the detector frame,
    f = f_u e1 + f_v e2, the change of variables (phi, f_u, f_v) has
    Jacobian ``-f_u cos(theta0)``: the ``|f_u|`` is absorbed by
    ``ramp_filter3`` and the ``cos(theta0)`` belongs to the inversion
    formula.  Dropping it inflates the reconstruction by exactly
    ``sec(theta0)`` (uniformly in space, so it is easy to miss).

    For ``theta0 != 0`` the data are intrinsically incomplete: the double
    cone of half-angle ``|theta0|`` about the rotation axis is *never*
    sampled, so the error plateaus as the number of projections grows
    instead of converging to zero.  Objects invariant along z live on
    ``f_z = 0`` and are therefore still recovered exactly.

    See doc/fbp3-filter-derivation.org, sections 5, 8 and Appendix A.
    """
    assert phi_axis.N == len(sinogram3)
    for p_uv_i in sinogram3:
        assert p_uv_i.shape == grid_uv.shape

    if radon_matrices:
        assert np.isclose(theta0.rad, 0.0), \
            "radon_matrices encode the untilted (theta = 0) geometry only"
        alpha = grid_uv.axis_x.T * grid_uv.axis_y.T / (axes3.axis_x.T * axes3.axis_y.T * axes3.axis_z.T)

    X_backproject = np.zeros(axes3.shape)
    for i, (phi_i, p_uv_i) in tenumerate(zip(phi_axis, sinogram3), total=phi_axis.N):
        grid_uv_ft_i, p_uv_ft_i = grid_uv.spectrum(p_uv_i, real=True)
        if i == 0:
            ramp = ramp_filter3(grid_uv_ft_i.Hz())
        p_uv_ft_ramp_i = p_uv_ft_i * ramp

        _, p_uv_filtered_i = grid_uv_ft_i.ispectrum(p_uv_ft_ramp_i)

        if radon_matrices:
            X_backproject_i = radon_matrices[i].T @ p_uv_filtered_i.flat
            X_backproject_i.shape = axes3.shape
            X_backproject += X_backproject_i * alpha
        else:
            X_backproject += backproject3(theta0, phi_i, axes3, grid_uv, p_uv_filtered_i)
    # The cos(theta0) is part of the inversion formula: the Jacobian of the
    # (phi, f_u, f_v) chart is -f_u cos(theta0).  See the docstring above.
    return phi_axis.rad.T * np.cos(theta0.rad) * X_backproject
