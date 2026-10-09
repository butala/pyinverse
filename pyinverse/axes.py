"""Regular grids in 3-D, and the VTK plumbing to display them.

Importing this module requires only NumPy and SciPy.  The rendering code
(:meth:`RegularAxes3.actor`, :meth:`RegularAxes3.volume`,
:meth:`RegularAxes3.voxel_actor`) resolves ``vtk`` and ``pyviz4d`` lazily, at
the point of use, so the numerical core works in a headless environment.
"""

import numpy as np
import scipy

from pyinverse._optional import optional_import

from .axis import Order, RegularAxis


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

    # https://github.com/paulo-herrera/PyEVTK

    # https://vtk.org/pipermail/vtkusers/2016-September/096626.html
    # VTK image data arrays are stored such that the X dimension
    # increases fastest, followed by Y, followed by Z.

    # Slow, middle, fast
    # row, column, depth
    # z,   y, ,    x
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
        return (f'<{self.__class__.__name__} <axis_x: {repr(self.axis_x)}> '
                f'<axis_y: {repr(self.axis_y)}> <axis_z {repr(self.axis_z)}>>')

    def __str__(self):
        return (f'{self.__class__.__name__}:\naxis x: {str(self.axis_x)}'
                f'\naxis y: {str(self.axis_y)}\naxis z: {str(self.axis_z)}')

    @property
    def shape(self):
        """
        """
        # row index changes slowest, then column, and depth changes fastest
        return tuple([axis.N for axis in [self.axis_z, self.axis_y, self.axis_x]])

    def __iter__(self):
        """
        Iterate over the grid center coordinates in storage order,
        i.e., z changes slowest, then y, and finally x changes
        fastest.
        """
        for i in self.axis_z:
            for j in self.axis_y:
                for k in self.axis_x:
                    yield (i, j, k)


    @property
    def centers(self):
        """
        """
        try:
            return self._centers
        except AttributeError:
            self._centers = np.meshgrid(self.axis_z.centers,
                                        self.axis_y.centers,
                                        self.axis_x.centers,
                                        indexing='ij')
            return self.centers

    @classmethod
    def unravel_indices(cls, indices, shape):
        """
        Convert the list of voxel flat *indices* to the tuple of
        list of indices in (z, y, x) order corresponding to a 3D axes
        with *shape* elements.
        """
        coords = np.unravel_index(indices, shape)
        return (coords[2], coords[1], coords[0])

    @classmethod
    def ravel_multi_index(cls, multi_index, dims):
        """
        Convert the tuple of list of indices in (z, y, x) order
        *multi_index* to an array of flattened indices corresponding
        to a 3D axes with *dims* shape.
        """
        assert len(multi_index) == len(dims) == 3
        assert len(multi_index[0]) == len(multi_index[1]) == len(multi_index[2])
        return np.ravel_multi_index([multi_index[2], multi_index[1], multi_index[0]], dims)

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
        f_axis_x = self.axis_x.spectrum_axis(s[2])
        f_axis_y = self.axis_y.spectrum_axis(s[1])
        f_axis_z = self.axis_z.spectrum_axis(s[0])
        return FreqRegularAxes3(f_axis_x, f_axis_y, f_axis_z, self)

    def spectrum(self, x, s=None):
        """
        """
        assert x.shape == self.shape
        assert (self.axis_x._order == Order.INCREASING
                and self.axis_y._order == Order.INCREASING
                and self.axis_z._order == Order.INCREASING)
        if s is None:
            s = self.shape
        elif s < self.shape:
            raise NotImplementedError()
        X_spectrum = scipy.fft.fftn(x, s=s)
        axes3_freq = self.spectrum_grid(s=s)
        P = np.exp(-1j*(axes3_freq.centers[2]*self.axis_x.x0 +
                        axes3_freq.centers[1]*self.axis_y.x0 +
                        axes3_freq.centers[0]*self.axis_z.x0))
        X_spectrum *= P * self.axis_x.T * self.axis_y.T * self.axis_z.T
        return axes3_freq, X_spectrum

    # The whole blanking system should be based on if X is a sparse
    # matrix. SciPy does not support sparse matrices of dimension
    # higher than 2, but the "sparse" package does. However, it
    # depends on numba which currently does not support python
    # 3.11. There is an update in the works.

    def _vtk_extent(self, cell=True):
        """``(origin, spacing)`` in VTK's (x, y, z) order.

        ``cell=True`` -- the cell grid :meth:`actor` / :meth:`volume` carry
        (one scalar per cell): the origin is the first *border*.
        ``cell=False`` -- the sample points of :meth:`to_vtk_image` (one
        scalar per sample): the origin is the first *centre*.  The two differ
        by half a cell, so the distinction matters.
        """
        axes = (self.axis_x, self.axis_y, self.axis_z)
        origin = tuple(ax.borders[0] if cell else ax.centers[0] for ax in axes)
        spacing = tuple(ax.T for ax in axes)
        return origin, spacing

    # Interesting discussion of VTK, LUT, and nan mapping
    # https://gitlab.kitware.com/vtk/vtk/-/issues/18197
    def _vtk_plot_setup(self, X, vmin=None, vmax=None, cmap='viridis', blank_nan=False):
        """
        """
        vtk = optional_import('vtk', extra='viz', purpose='RegularAxes3 rendering')
        numpy_support = optional_import('vtk.util.numpy_support', extra='viz',
                                        purpose='RegularAxes3 rendering')
        matplotlib_ctf = optional_import(
            'pyviz4d.volume', extra='viz',
            purpose='RegularAxes3 rendering').matplotlib_ctf
        assert X.shape == self.shape, (X.shape, self.shape)
        # A *fresh* grid on every call.  Caching one here made a second
        # actor()/volume() call rewrite the scalars of the first, silently
        # repainting an actor the caller was already holding; and a blanked
        # grid could not be reused at all (the old code raised AssertionError
        # to say so).  Fresh per call, both problems are gone.
        grid = vtk.vtkUniformGrid() if blank_nan else vtk.vtkImageData()
        Nz, Ny, Nx = self.shape
        grid.SetDimensions(Nx+1, Ny+1, Nz+1)
        origin, spacing = self._vtk_extent()
        grid.SetOrigin(*origin)
        grid.SetSpacing(*spacing)
        self._vtk_grid = grid          # the most recent grid, for callers that inspect it
        self._values = numpy_support.numpy_to_vtk(X.flat)
        grid.GetCellData().SetScalars(self._values)
        if blank_nan:
            for (k, j, i) in np.argwhere(np.isnan(X)):
                # VTK uses i=x, j=y, k=z for BlankCell.
                grid.BlankCell(i, j, k)
        if vmin is None:
            vmin = np.nanmin(X)
        if vmax is None:
            vmax = np.nanmax(X)
        # matplotlib_ctf(cmap_name, scalar_min, scalar_max) -> a
        # vtkColorTransferFunction over [vmin, vmax].
        self._lut = matplotlib_ctf(cmap, vmin, vmax)
        return vmin, vmax


    def actor(self, X, vmin=None, vmax=None, cmap='viridis', blank_nan=False):
        """Return a VTK actor for the 3-D array *X* on this grid.

        Requires the optional ``viz`` dependency
        (``pip install pyinverse[viz]``).
        """
        vtk = optional_import('vtk', extra='viz', purpose='RegularAxes3.actor')
        vmin, vmax = self._vtk_plot_setup(X, vmin=vmin, vmax=vmax, cmap=cmap, blank_nan=blank_nan)

        # Create a mapper and actor
        self._mapper = vtk.vtkDataSetMapper()
        self._mapper.SetInputData(self._vtk_grid)
        self._mapper.SetLookupTable(self._lut)
        self._mapper.SetScalarRange(vmin, vmax)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

        return self._actor


    def voxel_actor(self, ijk, color='CadetBlue'):
        """
        Create a VTK actor for the (*ijk*)th voxel element using
        web color *color*. Note that `i` corresponds to the z
        dimension, `j` to the y dimension, and `k` to the x dimension.
        Requires the optional ``viz`` dependency
        (``pip install pyinverse[viz]``).
        """
        # https://en.wikipedia.org/wiki/Web_colors
        vtk = optional_import('vtk', extra='viz',
                              purpose='RegularAxes3.voxel_actor')
        i, j, k = ijk

        xmin = self.axis_x.borders[k]
        xmax = self.axis_x.borders[k+1]

        ymin = self.axis_y.borders[j]
        ymax = self.axis_y.borders[j+1]

        zmin = self.axis_z.borders[i]
        zmax = self.axis_z.borders[i+1]

        cube = vtk.vtkCubeSource()
        cube.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)
        cube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(cube.GetOutput())

        colors = vtk.vtkNamedColors()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d(color))

        return actor


    # How to do this? https://www.kitware.com/volumetric-rendering-in-vtk-and-paraview-introducing-the-scattering-model-on-gpu/
    # https://www.kitware.com/cinematic-volume-rendering/
    def volume(self, X, vmin=None, vmax=None, cmap='viridis', amin=0, amax=1, blank_nan=False):
        """
        Return a VTK volume for the 3-D array *X* on this grid.  Requires the
        optional ``viz`` dependency (``pip install pyinverse[viz]``).
        """
        vtk = optional_import('vtk', extra='viz', purpose='RegularAxes3.volume')
        vmin, vmax = self._vtk_plot_setup(X, vmin=vmin, vmax=vmax, cmap=cmap, blank_nan=blank_nan)

        self._opacity_tf = vtk.vtkPiecewiseFunction()
        self._opacity_tf.AddPoint(vmin, amin)
        self._opacity_tf.AddPoint(vmax, amax)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(self._lut)
        volume_property.SetScalarOpacity(self._opacity_tf)
        #volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # vtkSmartVolumeMapper, not vtkOpenGLGPUVolumeRayCastMapper: the latter
        # needs OpenGL outright, while SmartVolumeMapper picks a back end (what
        # pyviz4d's VolumeActor does), so volume() also works on a headless box.
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(self._vtk_grid)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        return volume


    def to_vtk_image(self, X):
        """Return a point-sampled ``vtkImageData`` of *X* on this grid.

        This is the seam to `pyviz4d <https://github.com/butala/pyviz4d>`_:
        ``VolumeActor``, ``IsosurfaceActor`` and ``contour_actor`` all consume
        a ``vtkImageData``, and this is the packing they expect -- scalars at
        the sample points, in VTK's x-fastest order, which is exactly the
        C-order flattening of the ``(Nz, Ny, Nx)`` array.  Unlike
        :meth:`actor` / :meth:`volume`, which cache a *cell*-data grid on
        ``self``, the image is fresh on every call, so several of them can be
        handed to different actors without clashing.

        Requires the optional ``viz`` dependency
        (``pip install pyinverse[viz]``).
        """
        vtk = optional_import('vtk', extra='viz', purpose='RegularAxes3.to_vtk_image')
        numpy_support = optional_import('vtk.util.numpy_support', extra='viz',
                                        purpose='RegularAxes3.to_vtk_image')
        assert X.shape == self.shape
        Nz, Ny, Nx = self.shape
        image = vtk.vtkImageData()
        image.SetDimensions(Nx, Ny, Nz)
        # one point per sample: origin at the first sample *centre*, half a
        # cell in from the cell grid's corner (see `_vtk_extent`)
        origin, spacing = self._vtk_extent(cell=False)
        image.SetOrigin(*origin)
        image.SetSpacing(*spacing)
        values = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(X, dtype=np.float32).ravel(), deep=True,
            array_type=vtk.VTK_FLOAT)
        image.GetPointData().SetScalars(values)
        return image


    def isosurface_actor(self, X, levels, opacity=0.4, colors=None):
        """Return a VTK actor holding the isosurface(s) of *X* at *levels*.

        Flying-edges contouring (via ``pyviz4d.volume.contour_actor``) of
        :meth:`to_vtk_image`, so a reconstruction can be shown as translucent
        shells alongside the voxel :meth:`actor`, the direct :meth:`volume`, or
        the ground truth :meth:`~pyinverse.phantom3.Phantom3.actor`.  *colors*
        maps each level to an RGB triple; the default runs from dark purple to
        yellow (the two ends of Matplotlib's magma).

        Requires the optional ``viz`` dependency
        (``pip install pyinverse[viz]``).
        """
        # to_vtk_image resolves vtk; only pyviz4d's contouring is needed here
        contour_actor = optional_import(
            'pyviz4d.volume', extra='viz',
            purpose='RegularAxes3.isosurface_actor').contour_actor
        levels = list(levels)
        assert len(levels) >= 1
        _, _, actor = contour_actor(self.to_vtk_image(X), iso_values=levels,
                                    colors=colors, opacity=opacity)
        return actor


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

    X = np.array([[[1, 11, 21], [2, 12, 22], [3, 13, 23], [4, 14, 24]],
                  [[5, 15, 25], [6, 16, 26], [7, 17, 27], [8, 18, 28]]], dtype=float)
    Nz, Ny, Nx = X.shape

    axes3 = RegularAxes3.linspace((-1, 1.5, Nx),
                                  (-2, 3.5, Ny),
                                  (-3, 4, Nz))

    X[0, 0, 0] = np.nan
    X[0, 0, 1] = np.nan

    X_actor = axes3.actor(X, vmin=0, vmax=28, blank_nan=True)
    X_actor.GetProperty().LightingOff()

    # Translucent isosurface shells of the same field -- the view that reads a
    # 3-D reconstruction as structure rather than as a voxel cage.  (NaN
    # samples are simply outside every iso level, so the blanked cells drop
    # out of the shells.)  `to_vtk_image` is stateless, so it is safe to call
    # after `actor`.
    X_iso = axes3.isosurface_actor(X, levels=[5, 15, 25], opacity=0.5)

    # Direct volume rendering of the same field, as a composite blend.
    X_volume = axes3.volume(X, vmin=0, vmax=28, amin=0.0, amax=0.6)

    from pyviz4d import render_to_png

    from pyinverse.viz import show

    # One viewer for the pair -- add X_volume to the list for the composite
    # blend.  `show` puts a scalar bar on the colour map, writes the PNG and
    # opens the window on `--show`.
    show([X_actor, X_iso], png='/tmp/axes3_demo.png', scalar_bar=axes3._lut)

    # The windowless path, for the tests and any headless box.
    render_to_png([X_actor, X_iso], '/tmp/axes3_demo_offscreen.png')
