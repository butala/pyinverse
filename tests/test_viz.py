"""Tests for the 3-D rendering seam (``RegularAxes3.to_vtk_image`` and friends).

The whole ``viz`` stack is optional (``pip install pyinverse[viz]``), so
everything here skips when ``vtk`` is absent -- the numerical core must stay
importable and testable without it (see
``test_radon_calibration.test_importing_pyinverse_does_not_need_the_heavy_optionals``).

What is pinned:

* ``to_vtk_image`` packs the ``(Nz, Ny, Nx)`` array into a ``vtkImageData`` with
  the *sample* spacing and origin (one point per sample, x-fastest scalars --
  the C-order flattening), which is the convention ``pyviz4d``'s ``VolumeActor``
  / ``contour_actor`` consume;
* ``isosurface_actor`` produces a contour at the requested levels;
* the colour map comes from ``pyviz4d.volume.matplotlib_ctf`` and is a
  ``vtkColorTransferFunction`` usable as actor LUT and volume colour.
"""

import numpy as np
import pytest

vtk = pytest.importorskip('vtk')

from pyinverse.axes import RegularAxes3          # noqa: E402


@pytest.fixture
def axes3():
    return RegularAxes3.linspace((-1, 1, 5), (-2, 2, 6), (-3, 3, 7))


def test_to_vtk_image_has_the_sample_spacing_and_origin(axes3):
    image = axes3.to_vtk_image(np.zeros(axes3.shape))
    Nx, Ny, Nz = 5, 6, 7
    assert image.GetDimensions() == (Nx, Ny, Nz)
    assert image.GetOrigin() == (axes3.axis_x.centers[0],
                                 axes3.axis_y.centers[0],
                                 axes3.axis_z.centers[0])
    assert image.GetSpacing() == (axes3.axis_x.T,
                                  axes3.axis_y.T,
                                  axes3.axis_z.T)
    # one scalar per sample, not per cell (that is actor()/volume()'s model)
    assert image.GetPointData().GetScalars().GetNumberOfTuples() == Nx * Ny * Nz


def test_to_vtk_image_packs_scalars_x_fastest(axes3):
    """VTK's point index is ``i + j*Nx + k*Nx*Ny`` -- numpy C-order of the
    ``(Nz, Ny, Nx)`` array is exactly that, with no transpose anywhere."""
    Nz, Ny, Nx = axes3.shape
    X = np.arange(Nz * Ny * Nx, dtype=float).reshape(axes3.shape)
    image = axes3.to_vtk_image(X)
    values = image.GetPointData().GetScalars()
    for ijk in [(0, 0, 0), (0, 0, Nx - 1), (0, Ny - 1, 0), (Nz - 1, 0, 0),
                (Nz - 1, Ny - 1, Nx - 1), (2, 3, 4)]:
        k, j, i = ijk                      # (z, y, x) indexing of X
        vtk_index = i + j * Nx + k * Nx * Ny
        assert values.GetTuple1(vtk_index) == X[ijk], ijk


def test_to_vtk_image_rejects_the_wrong_shape(axes3):
    with pytest.raises(AssertionError):
        axes3.to_vtk_image(np.zeros((2, 2, 2)))


def test_isosurface_actor_contours_at_the_requested_levels(axes3):
    pyviz4d = pytest.importorskip('pyviz4d')       # the isosurface helper
    Nz, Ny, Nx = axes3.shape
    cz, cy, cx = (ax.centers for ax in
                  (axes3.axis_z, axes3.axis_y, axes3.axis_x))
    r = np.sqrt(cz[:, None, None]**2 + cy[None, :, None]**2 + cx[None, None, :]**2)
    X = np.exp(-(r / 0.6)**2)                      # a ball-ish blob

    actor = axes3.isosurface_actor(X, levels=[0.2, 0.5], opacity=0.3)
    assert actor.IsA('vtkActor')
    contour = actor.GetMapper().GetInputConnection(0, 0).GetProducer()
    # two iso levels in, two contour values out
    assert contour.GetNumberOfContours() == 2
    assert 0.0 < actor.GetProperty().GetOpacity() <= 1.0


def test_the_colour_map_is_a_vtk_colour_transfer_function(axes3):
    """The pyviz3d -> pyviz4d swap: ``matplotlib_ctf`` stands in for
    ``cmap2color_transfer_function`` and returns the same VTK type."""
    X = np.linspace(0.0, 1.0, np.prod(axes3.shape)).reshape(axes3.shape)
    vmin, vmax = axes3._vtk_plot_setup(X, cmap='viridis')
    assert (vmin, vmax) == (0.0, 1.0)
    assert axes3._lut.IsA('vtkColorTransferFunction')

    # and the LUT is accepted by both consumers: actor and volume
    actor = axes3.actor(X)
    assert actor.GetMapper().GetLookupTable() is axes3._lut
    vol = axes3.volume(X, amin=0.1, amax=0.9)
    assert vol.IsA('vtkVolume')


def test_a_fresh_image_per_call_does_not_clash(axes3):
    """actor()/volume() cache a grid on self; to_vtk_image must not."""
    X = np.zeros(axes3.shape)
    a = axes3.to_vtk_image(X)
    b = axes3.to_vtk_image(X + 1.0)
    assert a is not b
    assert a.GetPointData().GetScalars().GetTuple1(0) == 0.0
    assert b.GetPointData().GetScalars().GetTuple1(0) == 1.0
