# pyinverse

A lightweight inverse-problems and tomography testbed: coordinate axes and
grids, analytic phantoms, the 2-D Radon and 3-D ray transforms in matrix form,
and filtered backprojection.

## Install

```sh
pip install -e .
```

The numerical core depends on **NumPy and SciPy only**.  Everything else is
optional and imported lazily, at the point of use -- a headless install with no
display stack, no compiler and no image libraries works fine:

```sh
pip install -e ".[test]"        # + pytest, to run the test suite
pip install -e ".[view]"        # + matplotlib, for Grid.plot / Grid.imshow
pip install -e ".[image]"       # + imageio,    for RegularGrid.from_image
pip install -e ".[viz]"         # + VTK,        for RegularAxes3.actor / .volume
pip install -e ".[progress]"    # + tqdm,       for progress bars
```

Optional pieces that are absent degrade gracefully: progress bars become plain
iteration, and the rendering methods raise an `ImportError` that names the extra
to install.

### The `lasserre` C extension

`pip install -e .` also tries to build a small C extension (`lasserre`, sources
in `pyinverse/lasserre/`) that computes polytope volumes.  It is **optional**: a
missing compiler only produces a warning, and the shared library is loaded
lazily, on the first call to `pyinverse.volume.lasserre_vol`.  Nothing in the
analytic Radon path needs it.  To point the loader at a library built
elsewhere, set `PYINVERSE_LASSERRE_DIR`.  `pyinverse.volume.lasserre_available()`
reports whether it could be loaded.

(The 3-D rendering helpers also use [`pyviz3d`](https://github.com/butala/pyviz3d)
for the colour transfer function and the interactive `Renderer`; it is not on
PyPI, so install it from source if you want the notebooks' viewers.)

## Quick start

```python
import numpy as np
from pyinverse import RegularAxis, RegularGrid, radon_matrix

# image grid (rows = vertical), and a sinogram grid whose x-axis is the
# projection angle in degrees and whose y-axis is the detector coordinate
grid = RegularGrid.linspace((-1, 1, 64), (-1, 1, 64))
grid_y = RegularGrid.linspace((0, 180, 32), (-1.42, 1.42, 64))

R = radon_matrix(grid, grid_y, n_cpu=1)          # scipy.sparse.csc_matrix
A = np.zeros(grid.shape); A[32, 32] = 1.0        # an image
sinogram = (R @ A.ravel()).reshape(grid_y.shape)  # (detector, angle)

print(R.shape)          # (Na*Np, Nx*Ny) = (2048, 4096)
print(R.nnz / R.shape[0] / R.shape[1])
```

`Row l * Na + k` of `R` is detector sample `l` of angle `k`; `to_sinogram` /
`from_sinogram` and `to_image` / `from_image` convert between the flat vectors
`R` acts on and the arrays the rest of the package uses:

```python
from pyinverse import to_sinogram, from_image

sinogram = to_sinogram(R @ from_image(A), grid_y)   # (Np, Na)
```

`radon_matrix` defaults to one worker process per CPU.  With `n_cpu=1` it runs
in process -- no `__main__` guard, no start-up cost -- which is what you want in
tests and notebooks.  On platforms whose start method is `spawn` (macOS,
Windows), `n_cpu > 1` must be called from inside an `if __name__ == '__main__':`
guard, otherwise each worker re-imports the calling script and the call
recurses.

## Tests

```sh
python -m pytest                                # unit + calibration tests
python tests/fbp_validation.py                   # the 2-D / 3-D FBP harness
python -m pyinverse.volume                       # polytope-volume self-check
```

`tests/test_radon_calibration.py` pins the forward operator against an
independent per-pixel computation and against the analytic chord profile of a
uniform disk:

* `to_sinogram(R @ x)` agrees with the straightforward per-pixel loop
  `RegularGrid.sinogram` (which also pins the row order);
* the line integral through a centred unit-density disk reproduces the analytic
  chord length `2 sqrt(r^2 - t^2)` to the pixel-rasterisation floor;
* summing a projection with the detector spacing returns the image's total mass.

## Layout

```
pyinverse/    the installable package
  axis.py frequency.py angle.py      1-D coordinate axes
  grid.py axes.py                    2-D / 3-D grids
  rect.py util.py                    Fessler rectangle functions, special functions
  ellipse.py ellipsoid.py            analytic ellipse / ellipsoid projections
  phantom.py phantom3.py             Shepp-Logan and 3-D phantoms
  radon.py ray3.py                   the forward operators, in matrix form
  fbp.py fbp3.py                     filtered backprojection
  volume.py                          polytope volumes (+ the optional C extension)
  lasserre/                          the optional C extension
tests/        the test suite and the FBP validation harness
notebooks/    exploration notebooks
doc/          the 3-D FBP filter derivation
```

See the notebooks in `notebooks/` for worked examples of the 2-D and 3-D
pipelines.
