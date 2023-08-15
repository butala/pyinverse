import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import vtk

from .angle import Angle
from .ellipsoid import Ellipsoid
from .axis import RegularAxis
from .axes import RegularAxes3


"""
The ELLIPSOID_MATRIX copies from this project: https://github.com/tsadakane/sl3d

Copyright
=========
3-clause BSD License
Copyright 2021 SADAKANE, Tomoyuki
https://github.com/tsadakane/sl3d
"""

"""
1) 'kak_slaney':

The 3D Shepp-Logan head phantom. A is the relative density of water.
Ref:
[1] Kak AC, Slaney M, Principles of Computerized Tomographic Imaging, 1988. p.102
    http://www.slaney.org/pct/pct-errata.html


2) 'yu_ye_wang':

A variant of the Kak-Slaney phantom in which the contrast is improved for better
visual perception.
Ref:
[2] Yu H, Ye Y, Wang G, Katsevich-Type Algorithms for Variable Radius Spiral Cone-Beam CT
    Proceedings of the SPIE, Volume 5535, p. 550-557 (2004)


3) toft_schabel:

The geometry of this phantom is based on the 2D phantom shown in [3] and [4].
(Maybe this is the original Shepp-Logan head phantom?)
In [5], the intensities of the Shepp-Logan are modified
to yield higher contrast in the image.
It is known as 'Modified Shepp-Logan' of the `phantom` function of "Image Processing Toolbox" for MATLAB
In [6], it is extended to the 3D version. The parameters are as below.
The formula of geometry transfom for this option is the same as of [6] to reproduce the result,
while for other options, kak-slaney and yu-ye-wang, it is different.

Ref:
[3] Kak AC, Slaney M, Principles of Computerized Tomographic Imaging, 1988. p.55
[4] Jain, Anil K., Fundamentals of Digital Image Processing, Englewood Cliffs, NJ, Prentice Hall, 1989, p. 439.
[5] Toft P, The Radon Transform: Theory and Implementation, 1996.
[6] Matthias Schabel (2021). 3D Shepp-Logan phantom
    (https://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom),
    MATLAB Central File Exchange. Retrieved April 29, 2021.


4) 'kuoy':

From phantominator source code, where the following is given as a reference:

Koay, Cheng Guan, Joelle E. Sarlls, and Evren
Özarslan. "Three-dimensional analytical magnetic resonance imaging
phantom in the Fourier domain." Magnetic Resonance in Medicine: An
Official Journal of the International Society for Magnetic Resonance
in Medicine 58.2 (2007): 430-436.
"""


ELLIPSOID_MATRIX = {
    'kak_slaney':
    #            a        b        c      x0       y0      z0    phi1  phi2   phi3   A
    #        -------------------------------------------------------------------------
    np.array([[ 0.6900,  0.920,  0.900,  0.000,   0.000,   0.000,  0   ,  0,  0,  2.00],
              [ 0.6624,  0.874,  0.880,  0.000,   0.000,   0.000,  0   ,  0,  0, -0.98],
              [ 0.4100,  0.160,  0.210, -0.220,   0.000,  -0.250,  108 ,  0,  0, -0.02],
              [ 0.3100,  0.110,  0.220,  0.220,   0.000,  -0.250,  72  ,  0,  0, -0.02],
              [ 0.2100,  0.250,  0.500,  0.000,   0.350,  -0.250,  0   ,  0,  0,  0.02],
              [ 0.0460,  0.046,  0.046,  0.000,   0.100,  -0.250,  0   ,  0,  0,  0.02],
              [ 0.0460,  0.023,  0.020, -0.080,  -0.650,  -0.250,  0   ,  0,  0,  0.01],
              [ 0.0460,  0.023,  0.020,  0.060,  -0.650,  -0.250,  90  ,  0,  0,  0.01],
              [ 0.0560,  0.040,  0.100,  0.060,  -0.105,   0.625,  90  ,  0,  0,  0.02],
              [ 0.0560,  0.056,  0.100,  0.000,   0.100,   0.625,  0   ,  0,  0, -0.02]]),

    'yu_ye_wang':
    #            a      b       c      x0       y0      z0   phi1 phi2 phi3   A
    #        -----------------------------------------------------------------
    np.array([[ 0.6900,  0.920,  0.900,   0   ,   0    ,  0    , 0  , 0, 0,   1.0 ],
              [ 0.6624,  0.874,  0.880,   0   ,   0    ,  0    , 0  , 0, 0,  -0.8 ],
              [ 0.4100,  0.160,  0.210,  -0.22,   0    , -0.250, 108, 0, 0,  -0.2 ],
              [ 0.3100,  0.110,  0.220,   0.22,   0    , -0.25 , 72 , 0, 0,  -0.2 ],
              [ 0.2100,  0.250,  0.500,   0   ,   0.35 , -0.25 , 0  , 0, 0,   0.2 ],
              [ 0.0460,  0.046,  0.046,   0   ,   0.1  , -0.25 , 0  , 0, 0,   0.2 ],
              [ 0.0460,  0.023,  0.020,  -0.08,  -0.65 , -0.25 , 0  , 0, 0,   0.1 ],
              [ 0.0460,  0.023,  0.020,   0.06,  -0.65 , -0.25 , 90 , 0, 0,   0.1 ],
              [ 0.0560,  0.040,  0.100,   0.06,  -0.105,  0.625, 90 , 0, 0,   0.2 ],
              [ 0.0560,  0.056,  0.100,   0   ,   0.100,  0.625, 0  , 0, 0,  -0.2 ]]),

    'toft_schabel':
    #              a     b     c     x0      y0      z0    phi1  phi2   phi3   A
    #        -----------------------------------------------------------------
    np.array([[ 0.6900, 0.9200, 0.810,  0   ,  0     ,  0   ,   0,     0,     0,  1.0 ],
              [ 0.6624, 0.8740, 0.780,  0   , -0.0184,  0   ,   0,     0,     0, -0.8 ],
              [ 0.1100, 0.3100, 0.220,  0.22,  0     ,  0   , -18,     0,    10, -0.2 ],
              [ 0.1600, 0.4100, 0.280, -0.22,  0     ,  0   ,  18,     0,    10, -0.2 ],
              [ 0.2100, 0.2500, 0.410,  0   ,  0.35  , -0.15,   0,     0,     0,  0.1 ],
              [ 0.0460, 0.0460, 0.050,  0   ,  0.1   ,  0.25,   0,     0,     0,  0.1 ],
              [ 0.0460, 0.0460, 0.050,  0   , -0.1   ,  0.25,   0,     0,     0,  0.1 ],
              [ 0.0460, 0.0230, 0.050, -0.08, -0.605 ,  0   ,   0,     0,     0,  0.1 ],
              [ 0.0230, 0.0230, 0.020,  0   , -0.606 ,  0   ,   0,     0,     0,  0.1 ],
              [ 0.0230, 0.0460, 0.020,  0.06, -0.605 ,  0   ,   0,     0,     0,  0.1 ]]),

    'koay':
    #              a     b     c     x0      y0      z0    phi1  phi2 phi3   A
    #        ------------------------------------------------------------------------
    np.array([[ 0.69,   0.92,  0.9,    0,     0,     0,     0,   0,    0,  2   ],
              [ 0.6624, 0.874, 0.88,   0,     0,     0,     0,   0,    0, -0.8 ],
              [ 0.41,   0.16,  0.21,  -0.22,  0,    -0.25,  108, 0,    0, -0.2 ],
              [ 0.31,   0.11,  0.22,   0.22,  0,    -0.25,  72,  0,    0, -0.2 ],
              [ 0.21,   0.25,  0.5,    0,     0.35, -0.25,  0,   0,    0,  0.2 ],
              [ 0.046,  0.046, 0.046,  0,     0.1,  -0.25,  0,   0,    0,  0.2 ],
              [ 0.046,  0.023, 0.02,  -0.08, -0.65, -0.25,  0,   0,    0,  0.1 ],
              [ 0.046,  0.023, 0.02,   0.06, -0.65, -0.25,  90,  0,    0,  0.1 ],
              [ 0.056,  0.04,  0.1,    0.06, -0.105, 0.625, 90,  0,    0,  0.2 ],
              [ 0.056,  0.056, 0.1,    0,     0.1,   0.625, 0,   0,    0, -0.2 ]])
}


class Phantom3:
    def __init__(self,
                 ellipsoid_matrix=ELLIPSOID_MATRIX,
                 key='toft_schabel'):
        """
        """
        self._ellipsoids = [Ellipsoid(*(list(row[:6]) + [Angle(deg=x) for x in row[6:9]] + [row[9]])) for row in ellipsoid_matrix[key]]

    def __call__(self, x, y, z):
        """
        """
        return sum([e(x, y, z) for e in self._ellipsoids])

    def raster(self, axes3, D=4):
        """
        """
        assert D >= 1
        ax_hires = RegularAxis(axes3.axis_x.x0, axes3.axis_x.T / D, axes3.axis_x.N * D)
        ay_hires = RegularAxis(axes3.axis_y.x0, axes3.axis_y.T / D, axes3.axis_y.N * D)
        az_hires = RegularAxis(axes3.axis_z.x0, axes3.axis_z.T / D, axes3.axis_z.N * D)
        axes3_hires = RegularAxes3(ax_hires, ay_hires, az_hires)
        a_z, a_y, a_x = axes3_hires.centers
        x = self(a_x.flatten(), a_y.flatten(), a_z.flatten())
        x.shape = axes3_hires.shape
        ones = np.ones([D, D, D])
        return sp.ndimage.convolve(x, ones/D**3, mode='constant')[::D, ::D, ::D]

    def actor(self, opacity=0.2, cmap='viridis'):
        """
        """
        cm = plt.get_cmap(cmap)
        assembly = vtk.vtkAssembly()
        for e in self._ellipsoids:
            actor = e.actor()
            actor.GetProperty().SetColor(*cm(e.rho * 255)[:3])
            actor.GetProperty().SetOpacity(opacity)
            assembly.AddPart(actor)
        return assembly

    def proj(self, theta, phi, grid, Y=None):
        """
        """
        if Y is None:
            Y = np.zeros((grid.shape))
        for e in self._ellipsoids:
            e.proj(theta, phi, grid, Y=Y)
        return Y


if __name__ == '__main__':
    p = Phantom3()

    actor = p.actor()

    from pyviz3d.viz import Renderer
    ren = Renderer()
    ren.add_actor(actor)
    #ren.axes_on(actor.GetBounds())
    ren.axes_on((-1, 1, -1, 1, -1, 1))
    ren.reset_camera()

    ren.start()
