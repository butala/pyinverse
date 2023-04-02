from dataclasses import dataclass

import numpy as np
import scipy
import vtk


"""
Take a look here:

X-ray transform and 3D Radon transform for ellipsoids and tetrahedra
"""

@dataclass
class Ellipsoid:
    a: float
    b: float
    c: float

    x0: float
    y0: float
    z0: float

    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    rho: float


    def __post_init__(self):
        self.alpha_rad = np.radians(self.alpha_deg)
        self.beta_rad = np.radians(self.beta_deg)
        self.gamme_rad = np.radians(self.gamma_deg)

    @property
    def R_matrix(self):
        try:
            return self._R_matrix
        except AttributeError:
            self._R_matrix = scipy.spatial.transform.Rotation.from_euler('ZXZ',
                                                                         [self.alpha_deg, self.beta_deg, self.gamma_deg],
                                                                         degrees=True).as_matrix()
            return self.R_matrix


    def __call__(self, x, y, z):
        """
        """
        p_xyz = self.R_matrix.T @ (np.array([x - self.x0, y - self.y0, z - self.z0]))
        if (p_xyz[0] / self.a)**2 + (p_xyz[1] / self.b)**2 + (p_xyz[2] / self.c)**2 <= 1:
            return self.rho
        else:
            return 0

    def actor(self):
        """
        """
        # ellipsoid
        ellipsoid = vtk.vtkParametricEllipsoid()
        ellipsoid.SetXRadius(self.a)
        ellipsoid.SetYRadius(self.b)
        ellipsoid.SetZRadius(self.c)
        # source
        ellipsoid_source = vtk.vtkParametricFunctionSource()
        ellipsoid_source.SetParametricFunction(ellipsoid)
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(ellipsoid_source.GetOutputPort())
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # apply standard (Z-X-Z) Euler angle rotation
        actor.RotateZ(self.alpha_deg)
        actor.RotateX(self.beta_deg)
        actor.RotateZ(self.gamma_deg)
        actor.SetPosition(self.x0, self.y0, self.z0)
        return actor


if __name__ == '__main__':
    e = Ellipsoid(0.6900, 0.9200, 0.810, 0, 0, 0, 0, 0, 0, 1.0)

    from pyviz3d.viz import Renderer

    actor = e.actor()
    actor.GetProperty().SetColor(1, 0, 0)

    ren = Renderer()
    ren.add_actor(actor)
    ren.axes_on(actor.GetBounds())
    ren.reset_camera()

    ren.start()
