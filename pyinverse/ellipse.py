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
    min_x, max_x = sorted([min_x, max_x])
    min_y, max_y = sorted([min_y, max_y])
    return min_x, min_y, max_x, max_y


# COULD DO THIS BETTER --- FIND POINTS OF INTERSECTION, FIT A
# POLYNOMIAL DEPENDING ON INTERSECTION TYPE, AND THEN INTEGRATE
def integrate_indicator_function(indicator_fun, bounds, N=20):
    """
    """
    min_x, max_x, min_y, max_y = bounds
    X, Y = np.meshgrid(np.linspace(min_x, max_x, N),
                       np.linspace(min_y, max_y, N))
    return sum([indicator_fun((x, y)) for x, y in zip(X.flat, Y.flat)]) / N**2


# METHODS ARE TOO COMPLICATED!
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
        """
        Avoid this calculation unless requested.
        """
        try:
            return self._bounds
        except AttributeError:
            self._bounds = get_ellipse_bb(self.x0, self.y0, self.a, self.b, self.phi)
            return self.bounds

    def __call__(self, x, y):
        """
        """
        x_prime = x - self.x0
        y_prime = y - self.y0
        return ((x_prime*self.cos_phi + y_prime*self.sin_phi)**2 / self.a_sq + (y_prime*self.cos_phi - x_prime*self.sin_phi)**2 / self.b_sq <= 1) * self.A

    def raster(self, regular_grid, doall=False, A=None, N=20):
        """
        """
        if A is None:
            A = np.zeros(regular_grid.shape)
        # find nonzero rows and cols
        min_x, min_y, max_x, max_y = self.bounds
        try:
            J1 = max(np.argwhere(regular_grid.axis_x.bounds[:-1] >= min_x)[0][0] - 1, 0)
            J2 = min(np.argwhere(regular_grid.axis_x.bounds[1:] <= max_x)[-1][0] + 1, regular_grid.axis_x.N - 1)
            I1 = max(np.argwhere(regular_grid.axis_y.bounds[:-1] >= min_y)[0][0] - 1, 0)
            I2 = min(np.argwhere(regular_grid.axis_y.bounds[1:] <= max_y)[-1][0] + 1, regular_grid.axis_y.N - 1)
        except IndexError:
            # ellipse is outside the raster window --- return the 0 matrix
            return A

        if doall:
            J1 = 0
            J2 = regular_grid.axis_x.N - 1
            I1 = 0
            I2 = regular_grid.axis_y.N - 1

        # Determine if corners of each pixel are contained inside the ellipse.
        X, Y = np.meshgrid(regular_grid.axis_x.bounds[J1:J2+2] - self.x0,
                           (regular_grid.axis_y.bounds[I1:I2+2] - self.y0))
        D = (X*self.cos_phi + Y*self.sin_phi)**2 / self.a_sq + (Y*self.cos_phi - X*self.sin_phi)**2 / self.b_sq

        n_rows = A.shape[0]
        for index_i, i in enumerate(range(I1, I2 + 1)):
            for index_j, j in enumerate(range(J1, J2 + 1)):
                D_bounds_ij = [D[index_i, index_j],
                               D[index_i, index_j + 1],
                               D[index_i + 1, index_j],
                               D[index_i + 1, index_j + 1]]
                if (np.array(D_bounds_ij) <= 1).all():
                    # all 4 points bounding the pixel are contained inside the ellipse
                    A[(n_rows - 1) - i, j] += self.A
                elif (np.array(D_bounds_ij) <= 1).any():
                    # the pixel partially intersects with the ellipse
                    indicator_fun = lambda x: self(x[0], x[1])
                    bounds = [regular_grid.axis_x.bounds[j],
                              regular_grid.axis_x.bounds[j+1],
                              regular_grid.axis_y.bounds[i],
                              regular_grid.axis_y.bounds[i+1]]
                    A[(n_rows - 1) - i, j] += integrate_indicator_function(indicator_fun, bounds, N=N)

        # INSTEAD OF FLIPPED ROW INDEXING, CAN YOU INSTEAD USE
        # MESHGRID FLIP AS USED IN projection METHOD?

        # Why the flipped row indexing, i.e., why do we index via (n_rows
        # - 1) - i and not i? As the code is written, row 0 of A is
        # associated with element 0 of grid.axis_y --- the smallest
        # (bottom-most) vertical coordinate. However, the typical
        # convention is origin=upper (using the terminology adopted in the
        # documentation for the Matplotlib imshow function), i.e., that
        # element [0,0] corresponds to the upper left corner of the axis
        # (the alternative convention of origin=lower places element [0,0]
        # to the lower left corer of the axis).

        return A

    def fourier_transform(self, fx, fy):
        """
        """
        pass

    def fourier_transform_grid(self, regular_grid):
        """
        """
        pass

    # DOUBLE CHECK FLIP CONVENTION origin=upper!!!
    def projection(self, thetas, t_axis, rect=False, y=None):
        """
        """
        # EXPLAIN SHIFT!!!
        thetas_rad = np.radians((thetas + 90) % 360)
        if rect:
            pass
        else:
            # EXPLAIN FLIP!!!
            THETA, T = np.meshgrid(thetas_rad, t_axis.centers[::-1])
            gamma = np.arctan2(self.y0, self.x0)
            s = np.sqrt(self.x0**2 + self.y0**2)
            TAU = T - s * np.cos(gamma - THETA)
            BETA = THETA - self.phi_rad
            ALPHA = np.sqrt(self.a**2 * np.cos(BETA)**2 + self.b**2 * np.sin(BETA)**2)
            I = abs(TAU) <= ALPHA
            if y is None:
                y = np.zeros_like(THETA)
            y[I] += 2 * self.A * self.a * self.b / ALPHA[I]**2 * np.sqrt(ALPHA[I]**2 - TAU[I]**2)
            return y
