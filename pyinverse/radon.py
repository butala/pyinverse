import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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


def radon_matrix(grid, grid_y, a=0):
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

    theta_rad = np.radians(grid_y.axis_x)

    data = []
    indices = []
    indptr = [0]

    if a != 0:
        # The Nx == Ny requirement could be removed --- use quadrature
        # on the line case to approximate the beam integral or the
        # hyperplane approach.
        assert Nx == Ny
        alpha = 1 / Tx

    for i in tqdm(range(Ny)):
        for j in range(Nx):
            center_y, center_x = grid[i, j]
            column_count = 0
            for k, theta_k in enumerate(theta_rad):
                t_prime = grid_y.axis_y.centers - center_x * np.cos(theta_k) - center_y * np.sin(theta_k)
                if a == 0:
                    # line
                    p_theta_k = srect_2D_proj([theta_k], t_prime, 1/Tx, 1/Ty)
                    I_nz = np.nonzero(p_theta_k[:, 0])[0]
                    data_k = p_theta_k[I_nz, 0]
                else:
                    # beam (Nx == Ny)
                    p_theta_k = square_proj_conv_rect(theta_k, alpha * t_prime, a) / a / alpha
                    I_nz = np.nonzero(p_theta_k)[0]
                    data_k = p_theta_k[I_nz]
                data.extend(data_k)
                indices.extend(I_nz * Na + k)
                column_count += len(I_nz)
            indptr.append(indptr[-1] + column_count)
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
        assert n == m
        a = grid.axis_x.T / grid_y.axis_y.T
    else:
        assert n == m
        a = args.beam

    R = radon_matrix(grid, grid_y, a=a)

    scipy.sparse.save_npz(args.H_filename, R)


if __name__ == '__main__':
    sys.exit(main())
