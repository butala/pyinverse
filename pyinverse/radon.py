"""The 2-D Radon (line-integral) transform, in matrix form.

The forward operator is assembled as a sparse matrix by
:func:`radon_matrix`, whose row order and normalisation are documented in the
:class:`RadonOperator`-style notes there and pinned by the calibration tests in
``tests/test_radon_calibration.py``.  :func:`to_sinogram` / :func:`from_sinogram`
and :func:`to_image` / :func:`from_image` convert between the flat vectors the
matrix acts on and the ``(detector, angle)`` / ``(row, column)`` arrays the rest
of the package uses.
"""

import multiprocessing
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from itertools import product

import numpy as np
import scipy.sparse

from ._optional import tqdm
from .axis import RegularAxis
from .grid import RegularGrid
from .rect import rect_conv_radon_rect, srect_2D_proj
from .volume import volume_cal

__all__ = ["from_image", "from_sinogram", "radon_affine_scale", "radon_matrix",
           "radon_matrix_ij_analytic", "radon_matrix_ij_polytope",
           "radon_translate", "regular_grid2polytope", "theta_grid2half_planes",
           "to_image", "to_sinogram"]


def radon_translate(theta_rad, r, x0, y0):
    """Translation property of the Radon transform."""
    return r - x0 * np.cos(theta_rad) - y0 * np.sin(theta_rad)


def angle_pi(a, b):
    """
    The angle pi function defined in Fessler's book equation
    (3.2.16).
    """
    if a * b > 0:
        return np.arctan(b / a)
    elif b == 0:
        return 0
    elif a == 0 and b != 0:
        return np.pi/2
    elif a * b < 0:
        return np.arctan(b / a) + np.pi
    else:
        raise AssertionError()


def radon_affine_scale(theta_rad, r, alpha, beta):
    """The affine scaling property of the Radon transform."""
    if theta_rad == 0:
        a = beta
        b = 0
    elif theta_rad == np.pi/2:
        a = 0
        b = alpha
    else:
        a = beta*np.cos(theta_rad)
        b = alpha*np.sin(theta_rad)
    theta_prime = angle_pi(a, b)
    scale_factor = 1/np.hypot(a, b)
    r_prime = r * np.abs(alpha) * beta * scale_factor
    return theta_prime, r_prime, scale_factor


def regular_grid2polytope(grid, ij):
    """
    Return the polytope, i.e., A and b in the equation Ax <= b,
    that corresponds to (ij)th grid element.
    """
    A = [[-1,  0],
         [ 1,  0],
         [ 0, -1],
         [ 0,  1]]
    i, j = ij
    b = [-grid.axis_y.borders[i],
          grid.axis_y.borders[i + 1],
         -grid.axis_x.borders[j],
          grid.axis_x.borders[j + 1]]
    return A, b


def theta_grid2half_planes(grid_y, kl, rad=False):
    """
    Return the polytope, i.e., A and b in the equation Ax <= b,
    that corresponds to the strip of integration for the (*kl*)th
    element in the Radon transform where k is the theta index and ell is
    the projection axis coordinate index which are specified by
    *grid_y*. If *rad*, then angular coordinates are given in radians
    as opposed to degrees.
    """
    # This code could be greatly simplified. See the use of
    # scipy.spatial.transform.Rotation in
    # pyinverse.ray3.theta_grid2half_planes
    k, ell = kl
    theta_k = grid_y.axis_x.centers[k]
    if not rad:
        theta_k = np.radians(theta_k)
    theta_k %= 2*np.pi
    if theta_k == 0:
        A = [[0, -1],
             [0,  1]]
        b = [-grid_y.axis_y.borders[ell],
              grid_y.axis_y.borders[ell+1]]
    elif theta_k == np.pi:
        A = [[0, -1],
             [0,  1]]
        b = [-grid_y.axis_y.borders[-(ell+2)],
              grid_y.axis_y.borders[-(ell+1)]]
    elif theta_k == np.pi/2:
        A = [[-1, 0],
             [ 1, 0]]
        b = [-grid_y.axis_y.borders[ell],
              grid_y.axis_y.borders[ell+1]]
    elif theta_k == 3*np.pi/2:
        A = [[-1, 0],
             [ 1, 0]]
        b = [-grid_y.axis_y.borders[-(ell+2)],
              grid_y.axis_y.borders[-(ell+1)]]
    else:
        c_k = np.cos(theta_k)
        s_k = np.sin(theta_k)
        y1, x1 = np.array([s_k*grid_y.axis_y.borders[ell], c_k*grid_y.axis_y.borders[ell]])
        y2, x2 = np.array([s_k*grid_y.axis_y.borders[ell+1], c_k*grid_y.axis_y.borders[ell+1]])
        A = np.array([[-s_k, -c_k],
                      [ s_k,  c_k]])
        b = np.array([-y1*s_k - x1*c_k,
                       y2*s_k + x2*c_k])
    return A, b


def radon_matrix_ij_polytope(grid, grid_y, ij, a=0, rad=False):
    """
    """
    i, j = ij
    Np, Na = grid_y.shape

    data = []
    indices = []

    for ell in range(Np):
        for k in range(Na):
            A_t, b_t = theta_grid2half_planes(grid_y, (k, ell), rad=rad)
            A_grid, b_grid = regular_grid2polytope(grid, (i, j))
            A_lass = np.vstack((A_grid, A_t))
            b_lass = np.hstack((b_grid, b_t))
            p_theta_lk = volume_cal(6, 2, A_lass, b_lass) / grid_y.axis_y.T
            if not np.allclose(p_theta_lk, 0):
                data.append(p_theta_lk)
                indices.append(ell * Na + k)
    return data, indices


def radon_matrix_ij_analytic(grid, grid_y, ij, a=0):
    """
    """
    data = []
    indices = []

    Ny, Nx = grid.shape
    Np, Na = grid_y.shape

    Tx = grid.axis_x.T
    Ty = grid.axis_y.T

    i, j = ij
    center_y, center_x = grid[i, j]
    theta_rad = np.radians(grid_y.axis_x)

    for k, theta_k in enumerate(theta_rad):
        t_prime = grid_y.axis_y.centers - center_x * np.cos(theta_k) - center_y * np.sin(theta_k)
        if a == 0:
            # line
            p_theta_k = srect_2D_proj([theta_k], t_prime, 1/Tx, 1/Ty)
            I_nz = np.nonzero(p_theta_k[:, 0])[0]
            data_k = p_theta_k[I_nz, 0]
        else:
            # The measurement is the *average* of the line integral over a beam
            # of width ``a`` (a ray or detector element of finite aperture).
            # rect_conv_radon_rect returns the integral over the beam, so divide
            # by the beam width.  As ``a -> 0`` this tends to the delta-function
            # line integration above, and ``a = grid_y.axis_y.T`` (the default of
            # the command-line interface) reproduces the per-element bin average
            # -- i.e. the polytope path -- to the discretisation floor.
            p_theta_k = rect_conv_radon_rect(theta_k, t_prime, Tx, Ty, 1/a) / a
            I_nz = np.nonzero(p_theta_k)[0]
            data_k = p_theta_k[I_nz]
        data.extend(data_k)
        indices.extend(I_nz * Na + k)
    return data, indices


def radon_matrix(grid, grid_y, a=0, n_cpu=None, chunksize=8,
                 _radon_matrix_ij=radon_matrix_ij_analytic):
    """Matrix form of the Radon transform of an object on *grid*.

    The object is specified on *grid* (an image array of shape
    ``grid.shape == (Ny, Nx)``) and the projections on *grid_y*, whose x-axis
    carries the projection angles (in degrees) and whose y-axis carries the
    detector sample points.  The parameter *a* specifies the beam width: use
    ``a=0`` for delta-function line integration and ``a>0`` for the rect
    integration of a finite-width beam.

    Returns:
        scipy.sparse.csc_matrix: shape ``(Na*Np, Nx*Ny)``.  The unknowns are
        ordered as the C-order flattening of ``(Ny, Nx)``, i.e. column
        ``i*Nx + j`` corresponds to ``image[i, j]``; use :func:`from_image` /
        :func:`to_image` to convert.  Rows are blocked **detector-major**: row
        ``l * Na + k`` is detector sample ``l`` of angle ``k``
        (``grid_y.axis_y`` and ``grid_y.axis_x`` respectively); use
        :func:`from_sinogram` / :func:`to_sinogram` to convert.

    Normalisation: for ``a=0`` the entries of a row are the lengths of the
    intersection of the corresponding line with the grid cells, so for a
    piecewise-constant image ``x`` the result is the (unit-density) line
    integral; see ``tests/test_radon_calibration.py``.

    Notes:
        Parallelism uses :mod:`multiprocessing`.  With ``n_cpu=1`` the loop runs
        **in process**, which needs no ``__main__`` guard and has no start-up
        cost -- prefer it for small problems and in tests.  On platforms whose
        start method is ``spawn`` (macOS, Windows) ``n_cpu > 1`` must be called
        from inside a ``if __name__ == '__main__':`` guard, otherwise each
        worker re-imports the calling script and the call recurses.  *n_cpu*
        defaults to ``None``, meaning ``multiprocessing.cpu_count()``.
    """
    Ny, Nx = grid.shape
    Np, Na = grid_y.shape

    if n_cpu is None:
        n_cpu = multiprocessing.cpu_count()
    if n_cpu < 1:
        raise ValueError(f'n_cpu must be a positive integer or None, got {n_cpu!r}')

    data = []
    indices = []
    indptr = [0]

    radon_matrix_helper = partial(_radon_matrix_ij, grid, grid_y, a=a)
    ij = product(range(Ny), range(Nx))

    pool = None
    if n_cpu == 1:
        results = tqdm((radon_matrix_helper(ij_k) for ij_k in ij), total=Nx*Ny)
    else:
        pool = multiprocessing.Pool(n_cpu)
        results = tqdm(pool.imap(radon_matrix_helper, ij, chunksize), total=Nx*Ny)

    try:
        for data_ij, indices_ij in results:
            data.extend(data_ij)
            indices.extend(indices_ij)
            indptr.append(indptr[-1] + len(data_ij))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    H = scipy.sparse.csc_matrix((data, indices, indptr), shape=(Na*Np, Nx*Ny))
    return H


def to_sinogram(vector, grid_y):
    """Reshape a Radon-matrix row (or ``R @ x``) into a ``(Np, Na)`` sinogram.

    The rows of the matrix returned by :func:`radon_matrix` are blocked
    detector-major, so row ``l*Na + k`` is detector sample ``l`` of angle ``k``.
    This returns the corresponding array ``sinogram[l, k]``, indexable against
    ``grid_y.axis_y`` (detector) and ``grid_y.axis_x`` (angle).
    """
    Np, Na = grid_y.shape
    vector = np.asarray(vector)
    if vector.shape != (Np*Na,):
        raise ValueError(f'vector must have shape ({Np*Na},), got {vector.shape}')
    return vector.reshape(Np, Na)


def from_sinogram(sinogram):
    """Inverse of :func:`to_sinogram`: flatten a ``(Np, Na)`` array to a vector."""
    sinogram = np.asarray(sinogram)
    if sinogram.ndim != 2:
        raise ValueError(f'sinogram must be 2-D, got {sinogram.ndim} dimensions')
    return sinogram.reshape(-1)


def to_image(vector, grid):
    """Reshape an unknown-vector into an image array of shape ``grid.shape``.

    Columns of the matrix returned by :func:`radon_matrix` are ordered as the
    C-order flattening of ``(Ny, Nx)``.
    """
    Ny, Nx = grid.shape
    vector = np.asarray(vector)
    if vector.shape != (Ny*Nx,):
        raise ValueError(f'vector must have shape ({Ny*Nx},), got {vector.shape}')
    return vector.reshape(Ny, Nx)


def from_image(image):
    """Inverse of :func:`to_image`: flatten a ``(Ny, Nx)`` array to a vector."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f'image must be 2-D, got {image.ndim} dimensions')
    return image.reshape(-1)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Compute Radon transform matrix.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('H_filename',
                        type=str,
                        help='output matrix filename (in scipy.sparse npz format)')
    parser.add_argument('-n',
                        type=int,
                        required=True,
                        help='number of horizontal pixels')
    parser.add_argument('-m',
                        type=int,
                        required=False,
                        default=None,
                        help='number of vertical pixels (default to n if not specified)')
    parser.add_argument('--n_a',
                        '-a',
                        type=int,
                        required=False,
                        default=None,
                        help='number of angles (default to n if not specified)')
    parser.add_argument('--n_p',
                        '-p',
                        type=int,
                        required=False,
                        default=None,
                        help='number of projections (default to n if not specified)')
    parser.add_argument('--xlim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='horizontal axis bounds')
    parser.add_argument('--ylim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='vertical axis bounds')
    parser.add_argument('--tlim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='projection axis bounds')
    parser.add_argument('--beam',
                        type=float,
                        nargs='?',
                        required=False,
                        default=False,
                        help='')
    parser.add_argument('--n_cpu',
                        type=int,
                        required=False,
                        default=None,
                        help='number of worker processes; 1 runs in process '
                             '(no __main__ guard needed); default: all CPUs')
    args = parser.parse_args(argv[1:])

    n = args.n
    m = args.m if args.m is not None else n
    n_a = args.n_a if args.n_a is not None else n
    n_p = args.n_p if args.n_p is not None else n

    axis_x = RegularAxis.linspace(args.xlim[0], args.xlim[1], n)
    axis_y = RegularAxis.linspace(args.ylim[0], args.ylim[1], m)
    axis_t = RegularAxis.linspace(args.tlim[0], args.tlim[1], n_p)
    axis_theta = RegularAxis.linspace(0, 180, n_a, endpoint=False)

    grid = RegularGrid(axis_x, axis_y)
    grid_y = RegularGrid(axis_theta, axis_t)

    if args.beam is False:
        a = 0
    elif args.beam is None:
        a = 1 / grid_y.axis_y.T
    else:
        a = args.beam

    R = radon_matrix(grid, grid_y, a=a, n_cpu=args.n_cpu)

    scipy.sparse.save_npz(args.H_filename, R)


if __name__ == '__main__':
    sys.exit(main())
