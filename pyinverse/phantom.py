import numpy as np

from .ellipse import Ellipse


"""
Ellipse parameters for a 2-D phantom. The format is the same as used by the phantom function in Matlab. See https://www.mathworks.com/help/images/ref/phantom.html

Ellipses that define the phantom, specified as an e-by-6 numeric matrix defining e ellipses. The six columns of E are the ellipse parameters.

Column 1: A
Additive intensity value of the ellipse

Column 2: a
Length of the horizontal semiaxis of the ellipse

Column 3: b
Length of the vertical semiaxis of the ellipse

Column 4: x0
x-coordinate of the center of the ellipse

Column 5: y0
y-coordinate of the center of the ellipse

Column 6: phi
Angle (in degrees) between the horizontal semiaxis of the ellipse and the x-axis of the image


Parameters a, b, x0, and y0 are expressed in meters. The rotation angle is specified in degrees.
"""
ELLIPSE_MATRIX = {
    'modified-shepp-logan':
    #           rho   A      B      x1    y1     alpha (deg)
    #           (0)  (1)    (2)     (3)   (4)    (5)
    #          --------------------------------------------
    np.array([[  1, .69,   .92,     0,     0,     0 ],
              [-.8, .6624, .8740,   0,   -.0184,  0 ],
              [-.2, .1100, .3100,  .22,    0,    -18],
              [-.2, .1600, .4100, -.22,    0,     18],
              [ .1, .2100, .2500,   0,    .35,    0 ],
              [ .1, .0460, .0460,   0,    .1,     0 ],
              [ .1, .0460, .0460,   0,   -.1,     0 ],
              [ .1, .0460, .0230, -.08,  -.605,   0 ],
              [ .1, .0230, .0230,   0,   -.606,   0 ],
              [ .1, .0230, .0460,  .06,  -.605,   0 ]]),
    'shepp-logan':
    #           rho   A      B      x1    y1     alpha (deg)
    #           (0)  (1)    (2)     (3)   (4)    (5)
    #          --------------------------------------------
    np.array([[  1,  .69,   .92,     0,     0,     0 ],
              [-.98, .6624, .8740,   0,   -.0184,  0 ],
              [-.02, .1100, .3100,  .22,    0,    -18],
              [-.02, .1600, .4100, -.22,    0,     18],
              [ .01, .2100, .2500,   0,    .35,    0 ],
              [ .01, .0460, .0460,   0,    .1,     0 ],
              [ .01, .0460, .0460,   0,   -.1,     0 ],
              [ .01, .0460, .0230, -.08,  -.605,   0 ],
              [ .01, .0230, .0230,   0,   -.606,   0 ],
              [ .01, .0230, .0460,  .06,  -.605,   0 ]])
    }


class Phantom:
    def __init__(self,
                 ellipse_matrix=ELLIPSE_MATRIX,
                 key='modified-shepp-logan'):
        """
        """
        self._ellipses = [Ellipse(*row) for row in ellipse_matrix[key]]

    def __call__(self, x, y):
        """
        """
        return sum([e(x, y) for e in self._ellipses])

    def raster(self, regular_grid):
        """
        """
        A = np.zeros(regular_grid.shape)
        for e in self._ellipses:
            A = e.raster(regular_grid, A=A)
        return A

    def sinogram(self, sinogram_grid, rect=False, a=None):
        """
        """
        Y = np.zeros(sinogram_grid.shape)
        for e in self._ellipses:
            e.sinogram(sinogram_grid, rect=rect, a=a, Y=Y)
        return Y

    def proj_ft(self, sinogram_ft_grid, rect=False, a=None):
        """
        """
        Y_ft = np.zeros(sinogram_ft_grid.shape, dtype=np.complex)
        for e in self._ellipses:
            e.proj_ft(sinogram_ft_grid, rect=rect, a=a, Y_ft=Y_ft)
        return Y_ft
