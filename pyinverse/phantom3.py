import numpy as np
import matplotlib.pylab as plt
import vtk

from .ellipsoid import Ellipsoid


"""
From phantominator source code, where the following is given as a reference:

Koay, Cheng Guan, Joelle E. Sarlls, and Evren Özarslan.
           "Three‐dimensional analytical magnetic resonance imaging
           phantom in the Fourier domain." Magnetic Resonance in
           Medicine: An Official Journal of the International Society
           for Magnetic Resonance in Medicine 58.2 (2007): 430-436.
"""

ELLIPSE_MATRIX = {
    'modified-shepp-logan':
    #           rho   A      B     C      x1    y1      z1     alpha (deg)
    #           (0)  (1)    (2)   (3)     (4)   (5)     (6)    (7)
    #          -----------------------------------------------------------
    np.array([[  1, 0.69,   0.92,  0.9,   0,     0,     0,     0],
              [-.8, 0.6624, 0.874, 0.88,  0,     0,     0,     0],
              [-.2, 0.41,   0.16,  0.21, -0.22,  0,    -0.25,  3*np.pi/5],
              [-.2, 0.31,   0.11,  0.22,  0.22,  0,    -0.25,  2*np.pi/5],
              [ .1, 0.21,   0.25,  0.5,   0,     0.35, -0.25,  0],
              [ .1, 0.046,  0.046, 0.046, 0,     0.1,  -0.25,  0],
              [ .1, 0.046,  0.023, 0.02, -0.08, -0.65, -0.25,  0],
              [ .1, 0.046,  0.023, 0.02,  0.06, -0.65, -0.25,  np.pi/2],
              [ .1, 0.056,  0.04,  0.1,   0.06, -0.105, 0.625, np.pi/2],
              [ .1, 0.056,  0.056, 0.1,   0,     0.1,   0.625, 0]])
}


class Phantom3:
    def __init__(self,
                 ellipse_matrix=ELLIPSE_MATRIX,
                 key='modified-shepp-logan'):
        """
        """
        self._ellipsoids = [Ellipsoid(*row) for row in ellipse_matrix[key]]


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


if __name__ == '__main__':
    p = Phantom3()

    actor = p.actor()

    from pyviz3d.viz import Renderer
    ren = Renderer()
    ren.add_actor(actor)
    ren.axes_on(actor.GetBounds())
    ren.reset_camera()

    ren.start()
