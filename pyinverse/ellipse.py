from dataclasses import dataclass
import math

import numpy as np
import scipy.signal

from .util import robust_arcsin, robust_sqrt, besinc


# CLEANUP THINGS THAT USE np.tile --- BROADCASTING SHOULD BE SUFFICIENT

def ellipse_bb(x, y, major, minor, angle_deg):
    """
    Compute tight ellipse bounding box for the ellipse centered at (*x*, *y*), with major and minor axes lengths of *major* and *minor*, and rotated CCW by *angle_deg* (in degrees). Return the tuple `(min_x, min_y, max_x, max_y)`.

    Adapted from the following code: https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0

    The above code was written in response to a question: https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020
    """
    angle_deg %= 360
    angle_rad = np.radians(angle_deg)
    if angle_deg == 0:
        t = 0
    else:
        t = np.arctan(-minor * np.tan(angle_rad) / major)
    [max_x, min_x] = [x + major * np.cos(t) * np.cos(angle_rad) -
                      minor * np.sin(t) * np.sin(angle_rad) for t in (t, t + np.pi)]
    if angle_deg == 0:
        t = np.pi / 2
    else:
        t = np.arctan(minor * 1. / np.tan(angle_rad) / major)
    [max_y, min_y] = [y + minor * np.sin(t) * np.cos(angle_rad) +
                      major * np.cos(t) * np.sin(angle_rad) for t in (t, t + np.pi)]
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
    """ ??? """
    fx_rot = fx * ellipse.cos_phi + fy * ellipse.sin_phi
    fy_rot = fy * ellipse.cos_phi - fx * ellipse.sin_phi

    D = np.sqrt((ellipse.a * fx_rot)**2 + (ellipse.b * fy_rot)**2)

    P = np.exp(-2 * np.pi * 1j * (ellipse.x0 * fx + ellipse.y0 * fy))
    Z = besinc(2 * D) * 4 * P

    return Z * (ellipse.a * ellipse.b) * ellipse.rho


def ellipse_proj(ellipse, sinogram_grid, Y=None):
    """Calculate line integrals (from analytic expression) of *ellipse* at
    angles specified in degrees and projection axis sample points
    given by the x and y axes, respectively, of
    *sinogram_grid*. Return the resultant sinogram (# projections x #
    angles array). If *Y* is given, accumulate the sinogram in-place.

    """
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


# WHY ISN'T BEAM WIDTH A PARAMETER? Or, is this implicit and
# controllable in how axis_t is chosen?
def ellipse_proj_rect(ellipse, sinogram_grid, Y=None):
    """Calculate "beam" integrals (from analytic expression) of *ellipse*
    at angles specified in degrees and projection axis sample points
    given by the x and y axes, respectively, of
    *sinogram_grid*. Return the resultant sinogram (# projections x #
    angles array). If *Y* is given, accumulate the sinogram in-place.

    A "beam" integral is defined as follows. Let y(theta, t) equal the
    line integral of the ellipse at angle theta and projection axis
    coordinate t. The sinogram elements returned by
    :func:`ellipse_proj` are equal to y(theta_k, t_k). In contrast, a
    "beam" integral is given by

    y_beam(theta_k, t_k) = \int_{t_k - T/2}^{t_k + T/2} y(theta_k, \tau) d\tau

    where T is equal to the length between uniformly spaced projection
    axis sample points.

    """
    thetas_deg, axis_t = sinogram_grid.axis_x, sinogram_grid.axis_y

    na = len(thetas_deg)
    thetas_rad = np.radians(thetas_deg)

    t_center = axis_t.centers
    t_borders = axis_t.borders

    THETA, T = np.meshgrid(thetas_rad, t_center)

    ti = t_borders[:-1]
    ti.shape = axis_t.N, 1
    ti_plus_one = t_borders[1:]
    ti_plus_one.shape = axis_t.N, 1
    T1 = np.tile(ti, (1, na))
    T2 = np.tile(ti_plus_one, (1, na))

    phi_rad = math.radians(ellipse.phi_deg)
    ALPHA = THETA - phi_rad
    s = math.sqrt(ellipse.x0**2 + ellipse.y0**2)
    gamma = math.atan2(ellipse.y0, ellipse.x0)
    TAU = s * np.cos(gamma - THETA)

    Z = np.sqrt(ellipse.a**2 * np.cos(ALPHA)**2 + ellipse.b**2 * np.sin(ALPHA)**2)

    B1 = TAU - Z
    B2 = TAU + Z

    K = np.logical_and(T1 <= B2, T2 >= B1)

    T1[T1 < B1] = B1[T1 < B1]
    T2[T2 > B2] = B2[T2 > B2]

    A = Z[K]**2 - TAU[K]**2
    B = 2*TAU[K]
    C = -1

    DELTA = 4 * A * C - B**2

    J1 = -1 * robust_arcsin((2*C*T1[K] + B)/robust_sqrt(-DELTA))
    I1 = (2*C*T1[K] + B)*robust_sqrt(A + B*T1[K] + C*T1[K]**2)/(4*C) + DELTA/(8*C)*J1

    J2 = -1 * robust_arcsin((2*C*T2[K] + B)/robust_sqrt(-DELTA))
    I2 = (2*C*T2[K] + B)*robust_sqrt(A + B*T2[K] + C*T2[K]**2)/(4*C) + DELTA/(8*C)*J2
    if Y is None:
        Y = np.zeros((axis_t.N, na))
    Y[K] += ellipse.rho * ellipse.a * ellipse.b / Z[K]**2 * (I2 - I1) * axis_t.N

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
        Y_ft = np.zeros((nt, na), dtype=np.complex)
    Y_ft += ellipse_ft(ellipse, FX, FY)
    return Y_ft


def ellipse_proj_rect_ft(ellipse, sinogram_ft_grid, Y_ft=None):
    """ ??? """
    Y_ft = ellipse_proj_ft(ellipse, sinogram_ft_grid, Y_ft=Y_ft)
    Ts_t = sinogram_ft_grid.axis_y.axis_t.T
    alpha = 1 / (Ts_t / 2)
    W = np.sinc(sinogram_ft_grid.axis_y.centers / alpha)
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
    phi_deg: float  # Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

    """
    The notes on [length] and [mass] below concern the Shepp-Logan phantom.
    - A has units of density [mass] / [length]^2
    - a, b, x0, and y0 are given in the unit of [length], but in the relative sense (not in the absolute sense of, e.g., m)
    - phi_deg is in degrees
    """

    def __post_init__(self):
        self.a_sq = self.a**2
        self.b_sq = self.b**2
        self.phi_rad = np.radians(self.phi_deg)
        self.cos_phi = np.cos(self.phi_rad)
        self.sin_phi = np.sin(self.phi_rad)

    @property
    def bounds(self):
        """Return the ellipse bounding box. (See :func:`ellipse_bb`).

        """
        # Avoid bounding box calculation unless requested.
        try:
            return self._bounds
        except AttributeError:
            self._bounds = ellipse_bb(self.x0, self.y0, self.a, self.b, self.phi_deg)
            return self.bounds

    def __call__(self, x, y):
        """Evaluate the ellipse function at coordinates *x* and *y*
        (returning the ellipse density if *x* and *y* are interior to
        the ellipse and 0 otherwise).

        """
        x_prime = x - self.x0
        y_prime = y - self.y0
        return ((x_prime*self.cos_phi + y_prime*self.sin_phi)**2 / self.a_sq + (y_prime*self.cos_phi - x_prime*self.sin_phi)**2 / self.b_sq <= 1) * self.rho

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

    def sinogram(self, sinogram_grid, rect=False, Y=None):
        """Return the sinogram of the ellipse. If *rect*, use "beam"
        integration. Otherwise, use line integration. (see
        :func:`ellipse_proj` and :func:`ellipse_proj_rect`)

        """
        if rect:
            Y = ellipse_proj_rect(self, sinogram_grid, Y=Y)
        else:
            Y = ellipse_proj(self, sinogram_grid, Y=Y)
        return Y

    def proj_ft(self, sinogram_ft_grid, rect=False, Y_ft=None):
        """ ??? """
        if rect:
            Y_ft = ellipse_proj_rect_ft(self, sinogram_ft_grid, Y_ft=Y_ft)
        else:
            Y_ft = ellipse_proj_ft(self, sinogram_ft_grid, Y_ft=Y_ft)
        return Y_ft
