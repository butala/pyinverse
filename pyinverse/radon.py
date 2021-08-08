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
from .rect import srect_2D_proj, square_proj_conv_rect


def radon_translate(theta_rad, r, x0, y0):
    """Translation property of the Radon transform."""
    return r - x0 * np.cos(theta_rad) - y0 * np.sin(theta_rad)


def angle_pi(a, b):
    """The angle pi function defined in Fessler's book equation (3.2.16).

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


def radon_matrix_ij(grid, grid_y, ij, a=0):
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
            if Nx == Ny:
                # beam: square grid
                p_theta_k = Tx * square_proj_conv_rect(theta_k, t_prime / Tx, a * Tx)
            else:
                # bream: rectangular grid
                theta_prime, t_prime2, scale_factor = radon_affine_scale(theta_k, t_prime, 1/Tx, 1/Ty)
                a_prime = a * np.hypot(Tx*np.cos(theta_k), Ty*np.sin(theta_k))
                p_theta_k = scale_factor * square_proj_conv_rect(theta_prime, t_prime2, a_prime)
            I_nz = np.nonzero(p_theta_k)[0]
            data_k = p_theta_k[I_nz]
        data.extend(data_k)
        indices.extend(I_nz * Na + k)
    return data, indices


def radon_matrix(grid, grid_y, a=0, n_cpu=multiprocessing.cpu_count()):
    """Calculate the matrix form of the Radon transform for an object
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

    radon_matrix_helper = partial(radon_matrix_ij, grid, grid_y, a=a)

    with multiprocessing.Pool(n_cpu) as pool:
        for data_ij, indices_ij in tqdm(pool.imap(radon_matrix_helper, ij), total=Nx*Ny):
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
