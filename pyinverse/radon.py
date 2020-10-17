import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import multiprocessing

import numpy as np
import scipy.sparse

from .axis import RegularAxis

# REMOVE THIS DEPENDENCY!
from pyradon.radon_new import radon_matrix



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
                        #default=(-np.sqrt(2), np.sqrt(2)),
                        help='projection axis bounds')
    parser.add_argument('--n_cpu',
                        type=int,
                        required=False,
                        default=multiprocessing.cpu_count(),
                        help='number of computational cores')
    args = parser.parse_args(argv[1:])

    n = args.n
    m = args.m if args.m is not None else n
    n_a = args.n_a if args.n_a is not None else n
    n_p = args.n_p if args.n_p is not None else n


    axis_x = RegularAxis.linspace(args.xlim[0], args.xlim[1], n)
    axis_y = RegularAxis.linspace(args.ylim[0], args.ylim[1], m)
    axis_t = RegularAxis.linspace(args.tlim[0], args.tlim[1], n_p)
    axis_theta = RegularAxis.linspace(0, 180, n_a, endpoint=False)

    R = radon_matrix(axis_x.borders,
                     axis_y.borders,
                     axis_t.borders,
                     axis_theta.centers,
                     n_cpu=args.n_cpu)

    scipy.sparse.save_npz(args.H_filename, R)


if __name__ == '__main__':
    sys.exit(main())
