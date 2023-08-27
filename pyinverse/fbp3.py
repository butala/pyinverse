import numpy as np
import scipy as sp
from tqdm.contrib import tenumerate

from .angle import Angle


def ramp_filter3(grid_uv_ft_Hz):
    """
    """
    Cu_Hz, Cv_Hz = grid_uv_ft_Hz.centers
    return np.sqrt(Cu_Hz**2 + Cv_Hz**2)


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

    interp2d = sp.interpolate.RegularGridInterpolator((grid_uv.axis_y.centers, grid_uv.axis_x.centers), X, method=method, bounds_error=True)

    X_backproject = interp2d(np.array([v_backproject.flatten(), u_backproject.flatten()]).T)
    X_backproject.shape = axes3.shape

    return X_backproject


def fbp3_theta0(axes3, grid_uv, phi_axis, sinogram3, radon_matrices=None, theta0=Angle(deg=0)):
    """
    phi_axis: AngleRegularAxis
    """
    assert phi_axis.N == len(sinogram3)
    for p_uv_i in sinogram3:
        assert p_uv_i.shape == grid_uv.shape

    if radon_matrices:
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
    return phi_axis.rad.T * X_backproject
