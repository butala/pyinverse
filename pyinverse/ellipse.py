from dataclasses import dataclass
import math

import numpy as np
import scipy.signal


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



# DOCUMENT USE OF np.isclose INSTEAD OF math.isclose
def robust_scalar_arcsin(x):
    """ ??? """
    try:
        return math.asin(x)
    except ValueError:
        if x > 1:
            assert np.isclose(x, 1)
            return math.pi / 2
        if x < -1:
            assert np.isclose(x, -1)
            return -math.pi / 2
        raise

def robust_scalar_sqrt(x):
    """ ??? """
    try:
        return math.sqrt(x)
    except ValueError:
        if x < 0:
            assert np.isclose(x, 0)
            return 0.0
        raise

robust_arcsin = np.vectorize(robust_scalar_arcsin)
robust_sqrt = np.vectorize(robust_scalar_sqrt)


def ellipse_proj_rect(spec, thetas_deg, t_axis, Y=None):
    """ ??? """
    na = len(thetas_deg)

    thetas_rad = np.radians(thetas_deg)
    #t_center = fft_samples(np)
    #t_bound = get_bounds(t_center)

    # t_center = t_axis.centers
    # t_bound = t_axis.bounds

    # EXPLAIN FLIP!!!
    t_center = t_axis.centers[::-1]
    t_bound = t_axis.bounds[::-1]

    THETA, T = np.meshgrid(thetas_rad, t_center)

    ti = t_bound[:-1]
    ti.shape = t_axis.N, 1
    ti_plus_one = t_bound[1:]
    ti_plus_one.shape = t_axis.N, 1
    T1 = np.tile(ti, (1, na))
    T2 = np.tile(ti_plus_one, (1, na))
    T1, T2 = T2, T1

    #phi_rad = math.radians(spec.phi_deg)
    phi_rad = math.radians(spec.phi)
    ALPHA = THETA - phi_rad
    s = math.sqrt(spec.x0**2 + spec.y0**2)
    gamma = math.atan2(spec.y0, spec.x0)
    TAU = s * np.cos(gamma - THETA)

    Z = np.sqrt(spec.a**2 * np.cos(ALPHA)**2 + spec.b**2 * np.sin(ALPHA)**2)

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
        Y = np.zeros((t_axis.N, na))
    Y[K] += spec.A * spec.a * spec.b / Z[K]**2 * (I2 - I1) * t_axis.N

    return Y


# METHODS ARE TOO COMPLICATED!
# CHANGE phi FIELD TO phi_deg
# CHANGE A FIELD TO rho
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

    # RENAME thetas TO thetas_deg
    def projection(self, thetas, t_axis, rect=False, Y=None):
        """
        """
        if Y is None:
            Y = np.zeros((t_axis.N, len(thetas)))

        thetas_rad = np.radians(thetas % 360)

        if rect:
            ellipse_proj_rect(self, thetas, t_axis, Y=Y)
        else:
            # EXPLAIN FLIP!!!
            THETA, T = np.meshgrid(thetas_rad, t_axis.centers[::-1])
            gamma = np.arctan2(self.y0, self.x0)
            s = np.sqrt(self.x0**2 + self.y0**2)
            TAU = T - s * np.cos(gamma - THETA)
            BETA = THETA - self.phi_rad
            ALPHA = np.sqrt(self.a**2 * np.cos(BETA)**2 + self.b**2 * np.sin(BETA)**2)
            I = abs(TAU) <= ALPHA
            Y[I] += 2 * self.A * self.a * self.b / ALPHA[I]**2 * np.sqrt(ALPHA[I]**2 - TAU[I]**2)
        return Y
