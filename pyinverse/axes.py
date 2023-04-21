from itertools import product

import vtk
import numpy as np
import scipy

from pyviz3d.util import cmap2color_transfer_function
from pyviz3d.viz import Renderer

from .axis import RegularAxis, Order


class RegularAxes3:
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis
        axis_z (RegularAxis): height axis

    """
    # See the example in the __main__ function for an explanation of
    # the storage order in numpy for 3-D arrays. The depth dimension
    # (axis=2, the z axis) index changes the fastest, then the column
    # dimension (axis=1, the x axis), and then the row dimension
    # (axis=0, the y axis).
    def __init__(self, axis_x, axis_y, axis_z):
        """ ??? """
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z

    @classmethod
    def linspace(cls, linspace_x, linspace_y, linspace_z):
        """ ??? """
        return cls(*(RegularAxis.linspace(*x) for x in [linspace_x, linspace_y, linspace_z]))

    def __repr__(self):
        return f'<{self.__class__.__name__} <axis_x: {repr(self.axis_x)}> <axis_y: {repr(self.axis_y)}> <axis_z {repr(self.axis_z)}>>'

    def __str__(self):
        return f'{self.__class__.__name__}:\naxis x: {str(self.axis_x)}\naxis y: {str(self.axis_y)}\naxis z: {str(self.axis_z)}'

    @property
    def shape(self):
        """
        """
        # row index changes slowest, then column, and depth changes fastest
        return tuple([axis.N for axis in [self.axis_y, self.axis_x, self.axis_z]])

    @property
    def centers(self):
        """
        """
        try:
            return self._centers
        except AttributeError:
            self._centers = np.meshgrid(self.axis_x.centers, self.axis_y.centers, self.axis_z.centers)
            return self.centers

    def scale(self, Sx, Sy, Sz):
        """ ??? """
        return RegularAxes3(self.axis_x.scale(Sx), self.axis_y.scale(Sy), self.axis_z.scale(Sz))

    def Hz(self):
        """
        """
        return self.scale(1 / (2*np.pi), 1 / (2*np.pi), 1 / (2*np.pi))

    def increasing(self, x=None):
        """
        """
        if x is not None:
            assert x.shape == self.shape

        axes3 = RegularAxes3(self.axis_x.increasing(), self.axis_y.increasing(), self.axis_z.increasing())

        if x is not None:
            x = scipy.fft.fftshift(x)
            return axes3, x
        else:
            return axes3

    def spectrum_grid(self, s=None):
        """
        """
        if s is None:
            s = self.shape
        f_axis_x = self.axis_x.spectrum_axis(s[1])
        f_axis_y = self.axis_y.spectrum_axis(s[0])
        f_axis_z = self.axis_z.spectrum_axis(s[2])
        return FreqRegularAxes3(f_axis_x, f_axis_y, f_axis_z, self)

    def spectrum(self, x, s=None):
        """
        """
        assert x.shape == self.shape
        assert self.axis_x._order == Order.INCREASING and self.axis_y._order == Order.INCREASING and self.axis_z._order == Order.INCREASING
        if s is None:
            s = self.shape
        elif s < self.shape:
            raise NotImplementedError()
        X_spectrum = scipy.fft.fftn(x, s=s)
        axes3_freq = self.spectrum_grid(s=s)
        P = np.exp(-1j*(axes3_freq.centers[0]*self.axis_x.x0 +
                        axes3_freq.centers[1]*self.axis_y.x0 +
                        axes3_freq.centers[2]*self.axis_z.x0))
        X_spectrum *= P * self.axis_x.T * self.axis_y.T * self.axis_z.T
        return axes3_freq, X_spectrum

    def _vtk_plot_setup(self, X, vmin=None, vmax=None, cmap='viridis'):
        """
        """
        try:
            self._vtk_grid
        except AttributeError:
            self._vtk_grid = vtk.vtkImageData()
            Ny, Nx, Nz = self.shape
            self._vtk_grid.SetDimensions(Nx+1, Ny+1, Nz+1)
            self._vtk_grid.SetOrigin(self.axis_x.borders[0],
                                     self.axis_y.borders[0],
                                     self.axis_z.borders[0])
            self._vtk_grid.SetSpacing(self.axis_x.T,
                                      self.axis_y.T,
                                      self.axis_z.T)
        assert X.shape == self.shape
        # "magic" to switch from row major order to VTK's order
        #
        # row major (C) order: Z (depth) index changes fastest, then X (column) index is the next, and then Y (row)
        # VTK order: X index changes fastest, then Y, and then Z is slowest
        self._values = vtk.util.numpy_support.numpy_to_vtk(X.swapaxes(0, 1).ravel(order='F'))
        self._vtk_grid.GetCellData().SetScalars(self._values)
        if vmin is None:
            vmin = X.min()
        if vmax is None:
            vmax = X.max()
        self._lut = cmap2color_transfer_function(vmin=vmin, vmax=vmax, cmap=cmap)
        return vmin, vmax


    def actor(self, X, vmin=None, vmax=None, cmap='viridis'):
        """ ??? """
        self._vtk_plot_setup(X, vmin=vmin, vmax=vmax, cmap=cmap)
        # Create a mapper and actor
        self._mapper = vtk.vtkDataSetMapper()
        self._mapper.SetInputData(self._vtk_grid)
        self._mapper.SetLookupTable(self._lut)
        self._mapper.SetScalarRange(vmin, vmax)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

        return self._actor


    def volume(self, X, vmin=None, vmax=None, cmap='viridis', amin=0, amax=1):
        """
        """
        vmin, vmax = self._vtk_plot_setup(X, vmin=vmin, vmax=vmax, cmap=cmap)

        self._opacity_tf = vtk.vtkPiecewiseFunction()
        self._opacity_tf.AddPoint(vmin, amin)
        self._opacity_tf.AddPoint(vmax, amax)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(self._lut)
        volume_property.SetScalarOpacity(self._opacity_tf)
        #volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        volume_mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(self._vtk_grid)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        return volume


class FreqRegularAxes3(RegularAxes3):
    def __init__(self, axis_x, axis_y, axis_z, axes3_s):
        super().__init__(axis_x, axis_y, axis_z)
        self.axes3_s = axes3_s


if __name__ == '__main__':
    # Example based on: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays

    # Note that row-major (C) ordering means Z (depth) index changes
    # fastest, then X (column) index is the next, and then Y (row)
    # index is the slowest.

    # The example below represents the following:
    #
    # depth=0:
    # [1 2 3 4;
    #  5 6 7 8]
    #
    # depth=1:
    # [11 12 13 14;
    #  15 16 17 18]
    #
    # depth=2:
    # [21 22 23 24;
    #  25 26 27 28]

    X = np.array([[[1, 11, 21], [2, 12, 22], [3, 13, 23], [4, 14, 24]], [[5, 15, 25], [6, 16, 26], [7, 17, 27], [8, 18, 28]]])
    Ny, Nx, Nz = X.shape

    axes3 = RegularAxes3.linspace((-1, 1.5, Nx),
                                  (-2, 3.5, Ny),
                                  (-3, 4, Nz))

    X_actor = axes3.actor(X, vmin=0, vmax=28)
    X_actor.GetProperty().LightingOff()

    # X_volume = axes3.volume(X, vmin=0, vmax=Nx*Ny*Nz, amin=0.2)

    ren = Renderer()
    ren.depth_peeling_setup()
    ren.add_actor(X_actor)
    # ren.add_volume(X_volume)
    ren.axes_on(X_actor.GetBounds())
    ren.colorbar(axes3._lut)
    ren.reset_camera()

    ren.start()
