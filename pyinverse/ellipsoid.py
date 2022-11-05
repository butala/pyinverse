from dataclasses import dataclass

import numpy as np
import vtk


@dataclass
class Ellipsoid:
    rho: float
    a: float
    b: float
    c: float
    x0: float
    y0: float
    z0: float
    phi_rad: float

    def __post_init__(self):
        self.phi_deg = np.degrees(self.phi_rad)

    def __call__(self, x, y, z):
        """
        """
        assert False

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
        actor.RotateZ(self.phi_deg)
        actor.SetPosition(self.x0, self.y0, self.z0)
        return actor


if __name__ == '__main__':
    e = Ellipsoid(1, 0.69, 0.92, 0.9, 0, 0, 0, 0)

    from pyviz3d.viz import Renderer

    actor = e.actor()
    actor.GetProperty().SetColor(1, 0, 0)

    ren = Renderer()
    ren.add_actor(actor)
    ren.axes_on(actor.GetBounds())
    ren.reset_camera()

    ren.start()
