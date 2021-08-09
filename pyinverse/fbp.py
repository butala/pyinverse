import numpy as np

from .radon import radon_matrix as calc_radon_matrix


def ramp_filter(axis_omega):
    """Return the ramp filter (in the frequency domain) for the frequency
    coordinates specified in *axis_omega* in units of rad / s.

    """
    return np.abs(axis_omega.Hz().centers)


class BackProjector():
    """
    """
    def __init__(self, grid, grid_y, radon_matrix=None, **kwds):
        self.grid = grid
        self.grid_y = grid_y
        if radon_matrix is None:
            radon_matrix = calc_radon_matrix(grid, grid_y, **kwds)
        else:
            assert radon_matrix.shape == (np.prod(grid_y.shape), np.prod(grid.shape))
        self.radon_matrix = radon_matrix
        self.alpha = grid_y.axis_y.T / (grid.axis_x.T  * grid.axis_y.T)

    def __getitem__(self, index):
        """
        """
        assert index >= 0 and index < self.grid_y.axis_x.N
        return self.radon_matrix.T[:, index::self.grid_y.axis_x.N] * self.alpha


    def __matmul__(self, other):
        """
        """
        y = self.radon_matrix.T @ other
        return y * self.alpha


def fbp(grid, grid_y, sinogram, radon_matrix=None, **kwds):
    """Given the construction domain specification *grid*, the projection
    domain specification *grid_y*, and *sinogram* (x axis is theta and
    y axis is t), calculate and return the tomographic reconstruction
    by filtered back projection.

    """
    assert grid_y.shape == sinogram.shape
    if radon_matrix is not None:
        backprojector = BackProjector(grid, grid_y, radon_matrix=radon_matrix, **kwds)
    axis_theta = grid_y.axis_x
    axis_t = grid_y.axis_y
    grid_y_omega0, FT0_sinogram = grid_y.spectrum(sinogram, real=True, axis=0)
    axis_omega_t = grid_y_omega0.axis_y
    W = ramp_filter(axis_omega_t)
    # apply ramp filter to each column of the sinogram
    _, sinogram_ramp = grid_y_omega0.ispectrum(np.atleast_2d(W).T * FT0_sinogram, axis=0)
    S = np.zeros(grid.shape)
    if radon_matrix is None:
        X, Y = np.meshgrid(grid.axis_x.centers, grid.axis_y.centers)
    # process each projection for each angle
    for k, theta_k in enumerate(np.radians(axis_theta.centers)):
        # compute backprojection of the ramp filtered projection
        if radon_matrix is None:
            # calculate the backprojection by linear interpolation in the projection domain
            t_theta_k = X * np.cos(theta_k) + Y * np.sin(theta_k)
            S_k = np.interp(t_theta_k.flat, axis_t.centers, sinogram_ramp[:, k], left=np.nan, right=np.nan)
        else:
            # use Radon transform matrix if provided
            S_k = backprojector[k] @ sinogram_ramp[:, k]
        S_k.shape = S.shape
        # accumulate the result
        S += S_k
    S *= np.radians(axis_theta.T)
    return S
