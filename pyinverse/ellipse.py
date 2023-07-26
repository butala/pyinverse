from dataclasses import dataclass
import math

import numpy as np
import scipy.signal

from .angle import Angle
from .util import robust_arcsin, robust_sqrt, besinc
from .radon import radon_translate, radon_affine_scale


def ellipse_bb(x, y, major, minor, angle):
    """
    Compute tight ellipse bounding box for the ellipse centered at
    (*x*, *y*), with major and minor axes lengths of *major* and
    *minor*, and rotated CCW by :class:`Angle` *angle*. Return the
    tuple `(min_x, min_y, max_x, max_y)`.

    Adapted from the following code: https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0

    The above code was written in response to a question: https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020
    """
    if angle.deg == 0:
        t = 0
    else:
        t = np.arctan(-minor * angle.tan / major)
    [max_x, min_x] = [x + major * np.cos(t) * angle.cos -
                      minor * np.sin(t) * angle.sin for t in (t, t + np.pi)]
    if angle.deg == 0:
        t = np.pi / 2
    else:
        t = np.arctan(minor * 1. / angle.tan / major)
    [max_y, min_y] = [y + minor * np.sin(t) * angle.cos +
                      major * np.cos(t) * angle.sin for t in (t, t + np.pi)]
    min_x, max_x = sorted([min_x, max_x])
    min_y, max_y = sorted([min_y, max_y])
    return min_x, min_y, max_x, max_y


# Could do this better --- 1) find points of intersection, fit a
# polynomial depending on intersection type, and then integrate; 2)
# mesh and integrate; or 3) approximate as a simplex
def integrate_indicator_function(indicator_fun, bounds, N=20):
    """Calculate the definite integral of an indicator function (Boolean
    function with a sharp boundary). The indicator function
    *indicator_fun* takes two arguments (x and y coordinates) and
    returns a Boolean array (with the same shape as x and
    y). Integration bounds are specified in *bounds* using the
    `(min_x, min_y, max_x, max_y)` convention. Quadrature is simple:
    evaluate the indicator function over a *N*x*N* uniform grid
    spanning *bounds*.

    """
    min_x, max_x, min_y, max_y = bounds
    X, Y = np.meshgrid(np.linspace(min_x, max_x, N),
                       np.linspace(min_y, max_y, N))
    return sum([indicator_fun((x, y)) for x, y in zip(X.flat, Y.flat)]) / N**2



def ellipse_ft(ellipse, fx, fy):
    """Evaluate the 2D Fourier transform of *ellipse* at horizontal and
    vertical frequencies *fx* and *fy* (given in Hz).

    """
    fx_rot = fx * ellipse.phi.cos + fy * ellipse.phi.sin
    fy_rot = fy * ellipse.phi.cos - fx * ellipse.phi.sin

    D = np.sqrt((ellipse.a * fx_rot)**2 + (ellipse.b * fy_rot)**2)

    P = np.exp(-2 * np.pi * 1j * (ellipse.x0 * fx + ellipse.y0 * fy))
    Z = besinc(2 * D) * 4 * P

    return Z * (ellipse.a * ellipse.b) * ellipse.rho


def proj_disk(r, r0=1, alpha=1):
    """Projection of the disk function evaluated at coordinates *r*. The
    radius of the disk is *r0* and the magnitude (height) is *alpha*.

    """
    y = np.zeros_like(r)
    I = np.abs(r) <= r0
    y[I] = 2*alpha*np.sqrt(r0**2 - r[I]**2)
    return y


def ellipse_proj(ellipse, sinogram_grid, Y=None):
    """Calculate line integrals (from analytic expression) of *ellipse* at
    angles specified in degrees and projection axis sample points
    given by the x and y axes, respectively, of
    *sinogram_grid*. Return the resultant sinogram (# projections x #
    angles array). If *Y* is given, accumulate the sinogram in-place.

    """
    # Implementation using Radon transform properties.
    if Y is None:
        Y = np.zeros((sinogram_grid.shape))
    for k, theta_k in enumerate(np.radians(sinogram_grid.axis_x)):
        theta_prime = theta_k - ellipse.phi.rad
        t_prime = radon_translate(theta_k, sinogram_grid.axis_y.centers, ellipse.x0, ellipse.y0)
        theta_prime2, t_prime2, scale_factor = radon_affine_scale(theta_prime, t_prime, 1/ellipse.a, 1/ellipse.b)
        Y[:, k] += ellipse.rho * proj_disk(t_prime2) * scale_factor
    return Y


def ellipse_proj_direct(ellipse, sinogram_grid, Y=None):
    """Calculate line integrals (from analytic expression) of *ellipse* at
    angles specified in degrees and projection axis sample points
    given by the x and y axes, respectively, of
    *sinogram_grid*. Return the resultant sinogram (# projections x #
    angles array). If *Y* is given, accumulate the sinogram in-place.

    """
    # Implementation from formula given in Kak and Slaney.
    thetas_deg, axis_t = sinogram_grid.axis_x, sinogram_grid.axis_y
    thetas_rad = np.radians(thetas_deg)
    THETA, T = np.meshgrid(thetas_rad, axis_t.centers)
    gamma = np.arctan2(ellipse.y0, ellipse.x0)
    s = np.sqrt(ellipse.x0**2 + ellipse.y0**2)
    TAU = T - s * np.cos(gamma - THETA)
    BETA = THETA - ellipse.phi_rad
    ALPHA = np.sqrt(ellipse.a**2 * np.cos(BETA)**2 + ellipse.b**2 * np.sin(BETA)**2)
    I = abs(TAU) <= ALPHA
    if Y is None:
        Y = np.zeros((axis_t.N, len(thetas_deg)))
    Y[I] += 2 * ellipse.rho * ellipse.a * ellipse.b / ALPHA[I]**2 * np.sqrt(ALPHA[I]**2 - TAU[I]**2)
    return Y


def integral_sqrt_a2_minus_x2(x, a):
    """Integral of $\sqrt(a^2 - x^2)$ --- see (30) at
    http://integral-table.com.

    """
    return 0.5*x*np.sqrt(a**2 - x**2) + 0.5*a**2*np.arctan2(x, np.sqrt(a**2 - x**2))


def ellipse_proj_rect(ellipse, sinogram_grid, a, Y=None):
    """Calculate line integrals (from analytic expression) of *ellipse* at
    angles specified in degrees and projection axis sample points
    given by the x and y axes, respectively, of *sinogram_grid*
    convolved by $rect(a t)$. Return the resultant sinogram (#
    projections x # angles array). If *Y* is given, accumulate the
    sinogram in-place.

    """
    if Y is None:
        Y = np.zeros((sinogram_grid.shape))
    for k, theta_k in enumerate(np.radians(sinogram_grid.axis_x)):
        theta_prime = theta_k - ellipse.phi_rad
        t_prime = radon_translate(theta_k, sinogram_grid.axis_y.centers, ellipse.x0, ellipse.y0)
        theta_prime2, t_prime2, scale_factor = radon_affine_scale(theta_prime, t_prime, 1/ellipse.a, 1/ellipse.b)
        a_prime = a / scale_factor * ellipse.a * ellipse.b

        I = np.abs(t_prime2) < 1 + 1/(2*a_prime)
        t_prime2_left = t_prime2[I] - 1/(2*a_prime)
        t_prime2_left[t_prime2_left < -1] = -1
        t_prime2_right = t_prime2[I] + 1/(2*a_prime)
        t_prime2_right[t_prime2_right > 1] = 1

        I1 = integral_sqrt_a2_minus_x2(t_prime2_right, 1)
        I2 = integral_sqrt_a2_minus_x2(t_prime2_left, 1)

        Y[I, k] += 2*ellipse.rho*scale_factor*a_prime*(I1 - I2)
    return Y


def ellipse_proj_ft(ellipse, sinogram_ft_grid, Y_ft=None):
    """???"""
    thetas_deg, ft_axis = sinogram_ft_grid.axis_x, sinogram_ft_grid.axis_y
    na = len(thetas_deg)
    nt = len(ft_axis)

    theta_rads = np.radians(thetas_deg.centers)
    FX = np.atleast_2d(ft_axis.centers).T * np.atleast_2d(np.cos(theta_rads))
    FY = np.atleast_2d(ft_axis.centers).T * np.atleast_2d(np.sin(theta_rads))

    if Y_ft is None:
        Y_ft = np.zeros((nt, na), dtype=complex)
    Y_ft += ellipse_ft(ellipse, FX, FY)
    return Y_ft


def ellipse_proj_rect_ft(ellipse, sinogram_ft_grid, a, Y_ft=None):
    """ ??? """
    Y_ft = ellipse_proj_ft(ellipse, sinogram_ft_grid, Y_ft=Y_ft)
    Ts_t = sinogram_ft_grid.axis_y.axis_t.T
    W = np.sinc(sinogram_ft_grid.axis_y.centers / a)
    Y_ft *= np.atleast_2d(W).T
    return Y_ft


def ellipse_raster(ellipse, regular_grid, doall=False, A=None, N=20):
    """Return a rasterization, i.e., pixelization, of *ellipse* sampled at
    the center points of *regular_grid*. Skip a bounding box
    optimization if *doall* is set (the functionality has been
    validated, but the override remains in case there is an
    unconsidered edge case). If *A* is provided, accumulate the
    rasterization in-place. Use a resolution of *N* to calculate
    integrals over partial intersections (see
    :func:`integrate_indicator_function`).

    """
    if A is None:
        A = np.zeros(regular_grid.shape)
    # find nonzero rows and cols
    min_x, min_y, max_x, max_y = ellipse.bounds
    try:
        J1 = max(np.argwhere(regular_grid.axis_x.borders[:-1] >= min_x)[0][0] - 1, 0)
        J2 = min(np.argwhere(regular_grid.axis_x.borders[1:] <= max_x)[-1][0] + 1, regular_grid.axis_x.N - 1)
        I1 = max(np.argwhere(regular_grid.axis_y.borders[:-1] >= min_y)[0][0] - 1, 0)
        I2 = min(np.argwhere(regular_grid.axis_y.borders[1:] <= max_y)[-1][0] + 1, regular_grid.axis_y.N - 1)
    except IndexError:
        # ellipse is outside the raster window --- return the 0 matrix
        return A

    if doall:
        J1 = 0
        J2 = regular_grid.axis_x.N - 1
        I1 = 0
        I2 = regular_grid.axis_y.N - 1

    # Determine if corners of each pixel are contained inside the ellipse.
    X, Y = np.meshgrid(regular_grid.axis_x.borders[J1:J2+2] - ellipse.x0,
                       regular_grid.axis_y.borders[I1:I2+2] - ellipse.y0)
    D = (X*ellipse.cos_phi + Y*ellipse.sin_phi)**2 / ellipse.a_sq + (Y*ellipse.cos_phi - X*ellipse.sin_phi)**2 / ellipse.b_sq

    n_rows = A.shape[0]
    for i, A_i in enumerate(range(I1, I2 + 1)):
        for j, A_j in enumerate(range(J1, J2 + 1)):
            D_bounds_ij = [D[i, j], D[i, j + 1],
                           D[i + 1, j], D[i + 1, j + 1]]
            if (np.array(D_bounds_ij) <= 1).all():
                # all 4 points bounding the pixel are contained inside the ellipse
                A[A_i, A_j] += ellipse.rho
            elif (np.array(D_bounds_ij) <= 1).any():
                # the pixel partially intersects with the ellipse
                indicator_fun = lambda x: ellipse(x[0], x[1])
                bounds = [regular_grid.axis_x.borders[A_j],
                          regular_grid.axis_x.borders[A_j+1],
                          regular_grid.axis_y.borders[A_i],
                          regular_grid.axis_y.borders[A_i+1]]
                A[A_i, A_j] += integrate_indicator_function(indicator_fun, bounds, N=N)
    return A


@dataclass
class Ellipse:
    # Following convention in https://www.mathworks.com/help/images/ref/phantom.html
    rho: float      # Additive intensity value of the ellipse
    a: float        # Length of the horizontal semiaxis of the ellipse
    b: float        # Length of the vertical semiaxis of the ellipse
    x0: float       # x-coordinate of the center of the ellipse
    y0: float       # y-coordinate of the center of the ellipse
    phi: Angle      # Angle between the horizontal semiaxis of the ellipse and the x-axis of the image

    """
    The notes on [length] and [mass] below concern the Shepp-Logan phantom.
    - A has units of density [mass] / [length]^2
    - a, b, x0, and y0 are given in the unit of [length], but in the relative sense (not in the absolute sense of, e.g., m)
    - phi is an Angle (can be degrees or radians)
    """

    def __post_init__(self):
        self.a_sq = self.a**2
        self.b_sq = self.b**2

    @property
    def bounds(self):
        """Return the ellipse bounding box. (See :func:`ellipse_bb`).

        """
        # Avoid bounding box calculation unless requested.
        try:
            return self._bounds
        except AttributeError:
            self._bounds = ellipse_bb(self.x0, self.y0, self.a, self.b, self.phi)
            return self.bounds

    def __call__(self, x, y):
        """Evaluate the ellipse function at coordinates *x* and *y*
        (returning the ellipse density if *x* and *y* are interior to
        the ellipse and 0 otherwise).

        """
        x_prime = x - self.x0
        y_prime = y - self.y0
        return ((x_prime*self.phi.cos + y_prime*self.phi.sin)**2 / self.a_sq + (y_prime*self.phi.cos - x_prime*self.phi.sin)**2 / self.b_sq <= 1) * self.rho

    def raster(self, regular_grid, doall=False, A=None, N=20):
        """Return an image rasterization of the ellipse. (see
        :func:`ellipse_raster`).

        """
        return ellipse_raster(self, regular_grid, doall=doall, A=A, N=N)

    def fourier_transform(self, fx, fy):
        """
        """
        return ellipse_ft(self, fx, fy)

    def fourier_transform_grid(self, regular_grid):
        """
        """
        return ellipse_ft(self, *regular_grid.centers)

    def sinogram(self, sinogram_grid, rect=False, a=None, Y=None, _direct=False):
        """Return the sinogram of the ellipse. If *rect*, use "beam"
        integration. Otherwise, use line integration. (see
        :func:`ellipse_proj` and :func:`ellipse_proj_rect`)

        """
        if rect:
            assert a is not None
            Y = ellipse_proj_rect(self, sinogram_grid, a, Y=Y)
        else:
            if _direct:
                Y = ellipse_proj_direct(self, sinogram_grid, Y=Y)
            else:
                Y = ellipse_proj(self, sinogram_grid, Y=Y)
        return Y

    def proj_ft(self, sinogram_ft_grid, rect=False, a=None, Y_ft=None):
        """ ??? """
        if rect:
            assert a is not None
            Y_ft = ellipse_proj_rect_ft(self, sinogram_ft_grid, a, Y_ft=Y_ft)
        else:
            Y_ft = ellipse_proj_ft(self, sinogram_ft_grid, Y_ft=Y_ft)
        return Y_ft
