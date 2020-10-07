from dataclasses import dataclass

import numpy as np
import scipy.signal

from .grid import oversample_regular_grid


# https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
def get_ellipse_bb(x, y, major, minor, angle_deg):
    """
    Compute tight ellipse bounding box.

    MODIFIED: major and minor are the length of major and minor axes, NOT diameter

    see https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020

    From https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
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
    return min_x, min_y, max_x, max_y


@dataclass
class Ellipse:
    # Following convention in https://www.mathworks.com/help/images/ref/phantom.html
    A: float    # Additive intensity value of the ellipse
    a: float    # Length of the horizontal semiaxis of the ellipse
    b: float    # Length of the vertical semiaxis of the ellipse
    x0: float   # x-coordinate of the center of the ellipse
    y0: float   # y-coordinate of the center of the ellipse
    phi: float  # Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image

    def __post_init__(self):
        self.a_sq = self.a**2
        self.b_sq = self.b**2
        self.phi_rad = np.radians(self.phi)
        self.cos_phi = np.cos(self.phi_rad)
        self.sin_phi = np.sin(self.phi_rad)

    @property
    def bounds(self):
        try:
            return self._bounds
        except AttributeError:
            self._bounds = get_ellipse_bb(self.x0, self.y0, self.a, self.b, self.phi)
            return self.bounds

    def __call__(self, x, y):
        x_prime = x - self.x0
        y_prime = y - self.y0
        return ((x_prime*self.cos_phi + y_prime*self.sin_phi)**2 / self.a_sq + (y_prime*self.cos_phi - x_prime*self.sin_phi)**2 / self.b_sq <= 1) * self.A


def integrate_indicator_function(indicator_fun, bounds, N=20):
    """
    """
    min_x, max_x, min_y, max_y = bounds
    X, Y = np.meshgrid(np.linspace(min_x, max_x, N),
                       np.linspace(min_y, max_y, N))
    return sum([indicator_fun((x, y)) for x, y in zip(X.flat, Y.flat)]) / N**2


def raster_ellipse(e, regular_grid, oversample=2, doall=False):
    """
    """
    grid2 = oversample_regular_grid(regular_grid, oversample)
    A = np.zeros(grid2.shape)
    # find nonzero rows and cols
    min_x, min_y, max_x, max_y = e.bounds
    try:
        J1 = max(np.argwhere(grid2.axis_x.bounds[:-1] >= min_x)[0][0] - 1, 0)
        J2 = min(np.argwhere(grid2.axis_x.bounds[1:] <= max_x)[-1][0] + 1, grid2.axis_x.N - 1)
        I1 = max(np.argwhere(grid2.axis_y.bounds[:-1] >= min_y)[0][0] - 1, 0)
        I2 = min(np.argwhere(grid2.axis_y.bounds[1:] <= max_y)[-1][0] + 1, grid2.axis_y.N - 1)
    except IndexError:
        # ellipse is outside the raster window --- return the 0 matrix
        return A

    if doall:
        J1 = 0
        J2 = grid2.axis_x.N - 1
        I1 = 0
        I2 = grid2.axis_y.N - 1

    X, Y = np.meshgrid(grid2.axis_x.bounds[J1:J2+2] - e.x0,
                       (grid2.axis_y.bounds[I1:I2+2] - e.y0))
    D = (X*e.cos_phi + Y*e.sin_phi)**2 / e.a_sq + (Y*e.cos_phi - X*e.sin_phi)**2 / e.b_sq

    for index_i, i in enumerate(range(I1, I2 + 1)):
        for index_j, j in enumerate(range(J1, J2 + 1)):
            D_bounds_ij = [D[index_i, index_j],
                           D[index_i, index_j + 1],
                           D[index_i + 1, index_j],
                           D[index_i + 1, index_j + 1]]
            if (np.array(D_bounds_ij) <= 1).all():
                # all 4 points bounding the pixel are contained inside the ellipse
                A[i, j] += e.A * grid2.axis_x.T * grid2.axis_y.T
            elif (np.array(D_bounds_ij) <= 1).any():
                # the pixel partially intersects with the ellipse
                indicator_fun = lambda x: e(x[0], x[1])
                bounds = [grid2.axis_x.bounds[j],
                          grid2.axis_x.bounds[j+1],
                          grid2.axis_y.bounds[i],
                          grid2.axis_y.bounds[i+1]]
                I = integrate_indicator_function(indicator_fun, bounds)
                A[i, j] += I * e.A * grid2.axis_x.T * grid2.axis_y.T

    # Why the flip? A has been constructed row by row, starting from
    # row 0. As the code is written, row 0 of A is associated with
    # element 0 of grid.axis_y --- the smallest (bottom-most) vertical
    # coordinate. However, the typical convention is origin=upper
    # (using the terminology adopted in the documentation for the
    # Matplotlib imshow function), i.e., that element [0,0]
    # corresponds to the upper left corner of the axis (the
    # alternative convention of origin=lower places element [0,0] to
    # the lower left corer of the axis).

    return A[::-1, :]
