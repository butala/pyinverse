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

    @property
    def bounds(self):
        try:
            return self._bounds
        except AttributeError:
            self._bounds = get_ellipse_bb(self.x0, self.y0, self.a, self.b, self.phi)
            return self.bounds


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
        # ellipse is outside the raster window
        return A[::oversample, ::oversample]

    if doall:
        I1 = 0
        I2 = grid2.axis_y.N

        J1 = 0
        J2 = grid2.axis_x.N

    a_sq = e.a**2
    b_sq = e.b**2

    phi_rad = np.radians(e.phi)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    X, Y = np.meshgrid(grid2.axis_x[J1:J2+1] - e.x0,
                       (grid2.axis_y[I1:I2+1] - e.y0))

    D = (X * cos_phi + Y * sin_phi)**2 / a_sq + (Y * cos_phi - X * sin_phi)**2 / b_sq
    I = np.where(D <= 1)

    A[I1:I2+1, J1:J2+1][I] += e.A
    return A
