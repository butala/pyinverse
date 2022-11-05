from itertools import product

import vtk
import numpy as np

from pyviz3d.util import cmap2color_transfer_function
from pyviz3d.viz import Renderer

from .axis import RegularAxis


class RegularAxes:
    """
    """
    def __init__(self, *axis_list):
        self.axis_list = axis_list

    @classmethod
    def linspace(cls, *linspace_list):
        """ ??? """
        return cls(*[RegularAxis.linspace(*x) for x in linspace_list])

    def __repr__(self):
        return f'<{self.__class__.__name__} ' + ' '.join([f'<axis_{i}>: {repr(axis_i)}' for i, axis_i in enumerate(self.axis_list, 1)]) + '>'

    def __str__(self):
        return f'{self.__class__.__name__}:\n' + '\n'.join([f'axis {i}: {str(axis_i)}' for i, axis_i in enumerate(self.axis_list, 1)])

    def __iter__(self):
        return product(*self.axis_list)

    @property
    def shape(self):
        """
        """
        return tuple([axis.N for axis in self.axis_list[::-1]])


class RegularAxes3(RegularAxes):
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis
        axis_z (RegularAxis): height axis

    """
    def __init__(self, axis_x, axis_y, axis_z):
        """ ??? """
        super().__init__(axis_x, axis_y, axis_z)

    @property
    def axis_x(self):
        return self.axis_list[0]

    @property
    def axis_y(self):
        return self.axis_list[1]

    @property
    def axis_z(self):
        return self.axis_list[2]

    @classmethod
    def linspace(cls, linspace1, linspace2, linspace3):
        """ ??? """
        return super().linspace(linspace1, linspace2, linspace3)

    def __repr__(self):
        return f'<{self.__class__.__name__} <axis_x: {repr(self.axis_x)}> <axis_y: {repr(self.axis_y)}> <axis_z {repr(self.axis_z)}>>'

    def __str__(self):
        return f'{self.__class__.__name__}:\naxis x: {str(self.axis_x)}\naxis y: {str(self.axis_y)}\naxis z: {str(self.axis_z)}'

    def actor(self, X, vmin=None, vmax=None, cmap='viridis', **kwds):
        """ ??? """
        assert X.shape == self.shape
        try:
            self._vtk_grid
        except AttributeError:
            self._vtk_grid = vtk.vtkImageData()
            Nz, Ny, Nx = self.shape
            self._vtk_grid.SetDimensions(Nx+1, Ny+1, Nz+1)
            self._vtk_grid.SetOrigin(self.axis_x.borders[0],
                                     self.axis_y.borders[0],
                                     self.axis_z.borders[0])
            self._vtk_grid.SetSpacing(self.axis_x.T,
                                      self.axis_y.T,
                                      self.axis_z.T)
        self._values = vtk.util.numpy_support.numpy_to_vtk(X.flat)
        self._vtk_grid.GetCellData().SetScalars(self._values)
        if vmin is None:
            vmin = X.min()
        if vmax is None:
            vmax = X.max()
        # Create a mapper and actor
        self._mapper = vtk.vtkDataSetMapper()
        self._mapper.SetInputData(self._vtk_grid)
        self._lut = cmap2color_transfer_function(vmin=vmin, vmax=vmax, cmap=cmap)
        self._mapper.SetLookupTable(self._lut)
        self._mapper.SetScalarRange(vmin, vmax)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

        return self._actor


if __name__ == '__main__':
    Nx = 3
    Ny = 4
    Nz = 2

    axes3 = RegularAxes3.linspace((-1, 1.5, Nx),
                                  (-2, 3.5, Ny),
                                  (-3, 4, Nz))

    X = np.array(range(Nx*Ny*Nz))
    X.shape = axes3.shape

    X_actor = axes3.actor(X, vmin=0, vmax=Nx*Ny*Nz)
    X_actor.GetProperty().LightingOff()

    ren = Renderer()
    ren.add_actor(X_actor)
    ren.axes_on(X_actor.GetBounds())
    ren.colorbar(axes3._lut)
    ren.reset_camera()
    ren.start()
