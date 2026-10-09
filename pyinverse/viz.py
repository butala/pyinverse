"""Display helpers -- the one place the demos' viewer boilerplate lives.

Requires the optional ``viz`` dependency (``pip install pyinverse[viz]``):
VTK for the scalar bar and `pyviz4d <https://github.com/butala/pyviz4d>`_ for
the window itself.
"""

from __future__ import annotations

import sys

from ._optional import optional_import

__all__ = ["show"]


def show(actors, png=None, scalar_bar=None, interactive=None, size=(1200, 900)):
    """Assemble a `pyviz4d` viewer from *actors*, snapshot it, maybe open it.

    Args:
        actors: iterable of VTK props -- ``vtkActor``, ``vtkVolume``,
            ``vtkAssembly``, ...  Anything ``vtkRenderer.AddActor`` takes.
        png: if given, write a screenshot here first.  Needs no window, so a
            headless box still gets the picture.
        scalar_bar: if given, a lookup table (``vtkColorTransferFunction``)
            to draw as a legend down the right-hand edge.  ``Viewer4D`` shows
            an orientation triad of its own; a scalar field still wants a key.
        interactive: open the blocking window?  ``None`` means "when ``--show``
            is on the command line", so a demo's ``__main__`` needs no argument
            parsing of its own.
        size: window size in pixels.

    Returns:
        The configured ``pyviz4d.Viewer4D``.  ``start()`` has already run if
        *interactive*, so normally there is nothing left to do.

    Keys once the window is up: ``q`` quits, ``f`` toggles fullscreen, ``r``
    resets the camera to the view latched at ``start``.
    """
    from pyviz4d import Viewer4D        # optional: pyinverse[viz]

    ren = Viewer4D(size=size)
    for actor in actors:
        ren.add_actor(actor)
    if scalar_bar is not None:
        _add_scalar_bar(ren, scalar_bar)
    # else the camera opens *inside* the scene
    ren.ren.ResetCamera()
    if png is not None:
        ren.save_screenshot(png)
    if interactive is None:
        interactive = '--show' in sys.argv
    if interactive:
        ren.start()
    return ren


def _add_scalar_bar(ren, lut, n_labels=3):
    """Attach a legend for *lut* to the viewer's renderer."""
    vtk = optional_import('vtk', extra='viz', purpose='pyinverse.viz.show')
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetNumberOfLabels(n_labels)
    ren.ren.AddViewProp(bar)             # AddActor2D is gone as of VTK 9.7
