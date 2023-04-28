import sys
import multiprocessing
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from itertools import product

from tqdm import tqdm
import numpy as np
import scipy.sparse

from .axis import RegularAxis
from .grid import RegularGrid
from .rect import srect_2D_proj, rect_conv_radon_rect
from .volume import volume_cal


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
        assert False


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
    element in the Radon transform where k is the theta index and l is
    the projection axis coordinate index which are specified by
    *grid_y*. If *rad*, then angular coordinates are given in radians
    as opposed to degrees.
    """
    # This code could be greatly simplified. See the use of
    # scipy.spatial.transform.Rotation in
    # pyinverse.radon3.theta_grid2half_planes
    k, l = kl
    theta_k = grid_y.axis_x.centers[k]
    if not rad:
        theta_k = np.radians(theta_k)
    theta_k %= 2*np.pi
    if theta_k == 0:
        A = [[0, -1],
             [0,  1]]
        b = [-grid_y.axis_y.borders[l],
              grid_y.axis_y.borders[l+1]]
    elif theta_k == np.pi:
        A = [[0, -1],
             [0,  1]]
        b = [-grid_y.axis_y.borders[-(l+2)],
              grid_y.axis_y.borders[-(l+1)]]
    elif theta_k == np.pi/2:
        A = [[-1, 0],
             [ 1, 0]]
        b = [-grid_y.axis_y.borders[l],
              grid_y.axis_y.borders[l+1]]
    elif theta_k == 3*np.pi/2:
        A = [[-1, 0],
             [ 1, 0]]
        b = [-grid_y.axis_y.borders[-(l+2)],
              grid_y.axis_y.borders[-(l+1)]]
    else:
        c_k = np.cos(theta_k)
        s_k = np.sin(theta_k)
        y1, x1 = np.array([s_k*grid_y.axis_y.borders[l], c_k*grid_y.axis_y.borders[l]])
        y2, x2 = np.array([s_k*grid_y.axis_y.borders[l+1], c_k*grid_y.axis_y.borders[l+1]])
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

    for l in range(Np):
        for k in range(Na):
            A_t, b_t = theta_grid2half_planes(grid_y, (k, l), rad=rad)
            A_grid, b_grid = regular_grid2polytope(grid, (i, j))
            A_lass = np.vstack((A_grid, A_t))
            b_lass = np.hstack((b_grid, b_t))
            p_theta_lk = volume_cal(6, 2, A_lass, b_lass) / grid_y.axis_y.T
            if not np.allclose(p_theta_lk, 0):
                data.append(p_theta_lk)
                indices.append(l * Na + k)
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
            p_theta_k = rect_conv_radon_rect(theta_k, t_prime, Tx, Ty, 1/a) * a
            I_nz = np.nonzero(p_theta_k)[0]
            data_k = p_theta_k[I_nz]
        data.extend(data_k)
        indices.extend(I_nz * Na + k)
    return data, indices


def radon_matrix(grid, grid_y, a=0, n_cpu=multiprocessing.cpu_count(), chunksize=8,
                 _radon_matrix_ij=radon_matrix_ij_analytic):
    """
    Calculate the matrix form of the Radon transform for an object
    specified on *grid* and projections defined on *grid_y*. The
    parameter *a* specifies the beam width (rect integration applied
    to projections --- use delta function line integration when
    *a*=0).
    """
    Ny, Nx = grid.shape
    Np, Na = grid_y.shape

    Tx = grid.axis_x.T
    Ty = grid.axis_y.T

    data = []
    indices = []
    indptr = [0]

    ij = product(range(Ny), range(Nx))

    radon_matrix_helper = partial(_radon_matrix_ij, grid, grid_y, a=a)

    with multiprocessing.Pool(n_cpu) as pool:
        for data_ij, indices_ij in tqdm(pool.imap(radon_matrix_helper, ij, chunksize), total=Nx*Ny):
            data.extend(data_ij)
            indices.extend(indices_ij)
            indptr.append(indptr[-1] + len(data_ij))
    H = scipy.sparse.csc_matrix((data, indices, indptr), shape=(Na*Np, Nx*Ny))
    return H


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

    R = radon_matrix(grid, grid_y, a=a)

    scipy.sparse.save_npz(args.H_filename, R)


if __name__ == '__main__':
    sys.exit(main())
