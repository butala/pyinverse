import sys
import logging
import math
import multiprocessing
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from functools import partial

import numpy as np
import scipy as sp
import vtk
from tqdm import tqdm

from .grid import RegularGrid
from .axes import RegularAxes3
from .volume import volume_cal


def regular_axes2polytope(axes3, ijk):
    """
    Return the polytope, i.e., A and b in the equation Ax <= b,
    that corresponds to (*ijk*)th voxel element of *axes3*. The
    indexing convention is i, j, and k correspond to z, y, and x,
    respectively.
    """
    A = [[-1,  0,  0],
         [ 1,  0,  0],
         [ 0, -1,  0],
         [ 0,  1,  0],
         [ 0,  0, -1],
         [ 0,  0,  1]]
    i, j, k = ijk
    b = [-axes3.axis_z.borders[i],
          axes3.axis_z.borders[i + 1],
         -axes3.axis_y.borders[j],
          axes3.axis_y.borders[j + 1],
         -axes3.axis_x.borders[k],
         axes3.axis_x.borders[k + 1]]
    return A, b


def regular_axes2polytope2(axes3, i1, i2, j1, j2, k1, k2):
    """
    Return the polytope, i.e., A and b in the equation Ax <= b,
    that corresponds to portion of *axes3* with borders between the
    *i1*th and *i2*th, *j1*th and *j2*th, and *k1*th and *k2*th
    borders. The indexing convention is i, j, and k correspond to z, y,
    and x, respectively.
    """
    A = [[-1,  0,  0],
         [ 1,  0,  0],
         [ 0, -1,  0],
         [ 0,  1,  0],
         [ 0,  0, -1],
         [ 0,  0,  1]]
    b = [-axes3.axis_z.borders[i1],
          axes3.axis_z.borders[i2],
         -axes3.axis_y.borders[j1],
          axes3.axis_y.borders[j2],
         -axes3.axis_x.borders[k1],
          axes3.axis_x.borders[k2]]
    return A, b


def grid_uv2half_planes(theta, phi, grid_uv, mn, degrees=False):
    """
    Return the polytope, i.e., A and b in the equation Ax <= b,
    that corresponds to (*mn*)th u-v plane element of *grid_uv*. The
    angles *theta* and *phi* are the polar angle [-pi/2, pi/2] and
    azimuth [-pi, pi], respectively, specified in radians unless
    *degrees* is `True`.
    """
    A = np.array([[-1, 0,  0],
                  [ 1, 0,  0],
                  [ 0, 0, -1],
                  [ 0, 0,  1]])
    m, n = mn
    b = np.array([-grid_uv.axis_y.borders[m],
                   grid_uv.axis_y.borders[m+1],
                  -grid_uv.axis_x.borders[n],
                   grid_uv.axis_x.borders[n+1]])
    R = sp.spatial.transform.Rotation.from_euler('XZ', [phi, theta], degrees=degrees).as_matrix()
    return A @ R, b


def beam2actor(grid, ij, e_min_max, theta, phi, color='Peru', alpha=0.2, deg=False):
    """
    Return a VTK actor for the beam bounded by the (*ij*)th
    element of the u-v plane interval defined in *grid* where `i`
    corresponds to the vertical (v) dimension and `j` corresponds to
    the horizontal (u) dimension. The length of the beam (depth
    dimension) is specified by the tuple *e_min_max*. The bream is
    oriented by the angles *theta* and *phi* are the polar angle
    [-pi/2, pi/2] and azimuth [-pi, pi], respectively, specified in
    radians unless *degrees* is `True`. Use the web color *color* and
    opacity *alpha*.
    """
    i, j = ij
    emin, emax = e_min_max
    if deg:
        theta_deg = theta
        phi_deg = phi
    else:
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

    umin = grid.axis_x.borders[j]
    umax = grid.axis_x.borders[j+1]

    vmin = grid.axis_y.borders[i]
    vmax = grid.axis_y.borders[i+1]

    cube = vtk.vtkCubeSource()
    cube.SetBounds(umin, umax, emin, emax, vmin, vmax)
    cube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cube.GetOutput())

    colors = vtk.vtkNamedColors()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetOpacity(alpha)
    actor.RotateX(theta_deg)
    actor.RotateZ(phi_deg)

    return actor


def ray_row(A_mn, b_mn, u_T, v_T, axes3):
    """
    Divide and conquer approach to calculate the volume
    intersections of the parallel beam determined by the *A_mn* x <=
    *b_mn* (see `grid_uv2half_planes` to calculate *A_mn* and *b_mn*)
    with the voxels in the `RegularAxes3` *axes3*. The scalars *u_T*
    and *v_T* are width and height of the parallel beam in the u-v
    plane. Return a tuple with a list of nonzero volumes and the
    corresponding list of flat indices to the voxels.
    """
    Nz, Ny, Nx = axes3.shape

    data = []
    indices = []

    def ray_helper(data_i, indices_i, i1, i2, j1, j2, k1, k2):
        if (i2 <= i1) or (j2 <= j1) or (k2 <= k1):
            return

        A_ijk, b_ijk = regular_axes2polytope2(axes3, i1, i2, j1, j2, k1, k2)

        A_lass = np.vstack((A_ijk, A_mn))
        b_lass = np.hstack((b_ijk, b_mn))

        vol = volume_cal(10, 3, A_lass, b_lass) / (u_T * v_T)

        if np.allclose(vol, 0):
            return

        if (i2 == i1 + 1) and (j2 == j1 + 1) and (k2 == k1 + 1):
            data_i.append(vol)
            indices_i.append((k1, j1, i1))

        else:
            bi = i2 - i1
            ci = math.ceil(bi/2) + i1

            bj = j2 - j1
            cj = math.ceil(bj/2) + j1

            bk = k2 - k1
            ck = math.ceil(bk/2) + k1

            ray_helper(data_i, indices_i, i1, ci, j1, cj, k1, ck)
            ray_helper(data_i, indices_i, ci, i2, j1, cj, k1, ck)
            ray_helper(data_i, indices_i, i1, ci, cj, j2, k1, ck)
            ray_helper(data_i, indices_i, ci, i2, cj, j2, k1, ck)

            ray_helper(data_i, indices_i, i1, ci, j1, cj, ck, k2)
            ray_helper(data_i, indices_i, ci, i2, j1, cj, ck, k2)
            ray_helper(data_i, indices_i, i1, ci, cj, j2, ck, k2)
            ray_helper(data_i, indices_i, ci, i2, cj, j2, ck, k2)
        return data_i, indices_i

    data, ijk = ray_helper([], [], 0, Nz, 0, Ny, 0, Nx)
    flat_indices = RegularAxes3.ravel_multi_index(list(zip(*ijk)), axes3.shape)
    sorted_indices, sorted_data = list(zip(*sorted(zip(flat_indices, data), key=lambda x: x[0])))
    return sorted_data, sorted_indices


def ray_row_mn(theta, phi, axes3, grid_uv, mn, degrees=True):
    """ """
    A_mn, b_mn = grid_uv2half_planes(theta, phi, grid_uv, mn, degrees=degrees)
    return ray_row(A_mn, b_mn, grid_uv.axis_x.T, grid_uv.axis_y.T, axes3)


def ray_matrix(theta, phi, axes3, grid_uv, n_cpu=multiprocessing.cpu_count(), degrees=True):
    """
    """
    Nv, Nu = grid_uv.shape

    ij = product(range(Nv), range(Nu))

    ray_row_helper = partial(ray_row_mn, theta, phi, axes3, grid_uv, degrees=degrees)

    data = []
    indices = []
    indptr = [0]

    with multiprocessing.Pool(n_cpu) as pool:
        for data_mn, indices_mn in tqdm(pool.imap(ray_row_helper, ij), total=Nv*Nu):
            data.extend(data_mn)
            indices.extend(indices_mn)
            indptr.append(indptr[-1] + len(data_mn))
    H = sp.sparse.csr_matrix((data, indices, indptr), shape=[Nu * Nv, np.prod(axes3.shape)])
    return H


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Compute the 3-D ray transform matrix.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('H_filename',
                        type=str,
                        help='output matrix filename (in scipy.sparse npz format)')
    parser.add_argument('theta',
                        type=float,
                        help='polar angle ([-90, 90] degrees, rotation relative to the x-y plane)')
    parser.add_argument('phi',
                        type=float,
                        help='azimuth angle ([-180, 180] degrees, rotation about \hat{z})')
    xyz_group = parser.add_mutually_exclusive_group(required=True)
    xyz_group.add_argument('-n',
                           type=int,
                           help='axes3 Nx = Ny = Nz = n')
    xyz_group.add_argument('--xyz',
                           nargs=3,
                           type=int,
                           help='axes3 Nx, Ny, and Nz')
    uv_group = parser.add_mutually_exclusive_group(required=True)
    uv_group.add_argument('-u',
                          type=int,
                          help='uv-plane grid Nu = Nv = u')
    uv_group.add_argument('--uv',
                          nargs=2,
                          type=int,
                          help='uv-plane grid Nu and Nv')
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
    parser.add_argument('--zlim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='depth axis bounds')
    parser.add_argument('--ulim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='uv-plane horizontal axis bounds')
    parser.add_argument('--vlim',
                        type=float,
                        nargs=2,
                        default=(-1, 1),
                        help='uv-plane vertical axis bounds')
    parser.add_argument('--radians',
                        action='store_true',
                        help='theta and phi are specified in radians instead of degrees')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help='number of cores to use (note there is a limit of 61 on windows)')

    args = parser.parse_args(argv[1:])

    if args.radians:
        theta_deg = np.degrees(args.theta)
        phi_deg = np.degrees(args.phi)
    else:
        theta_deg = args.theta
        phi_deg = args.phi

    assert -90 <= args.theta <= 90
    assert -180 <= args.phi <= 180

    if args.n is not None:
        assert args.n > 0
        Nx = Ny = Nz = args.n
    else:
        for x in args.xyz:
            assert x > 0
        Nx, Ny, Nz = args.xyz

    if args.u is not None:
        assert args.u > 0
        Nu = Nv = args.u
    else:
        Nu, Nv = args.uv
        assert Nu > 0
        assert Nv > 0

    assert args.n_cpu > 0

    axes3 = RegularAxes3.linspace((args.xlim[0], args.xlim[1], Nx),
                                  (args.ylim[0], args.ylim[1], Ny),
                                  (args.zlim[0], args.zlim[1], Nz))

    grid_uv = RegularGrid.linspace((args.ulim[0], args.ulim[1], Nu),
                                   (args.vlim[0], args.vlim[1], Nz))

    H = ray_matrix(theta_deg, phi_deg, axes3, grid_uv, n_cpu=args.n_cpu)

    sp.sparse.save_npz(args.H_filename, H)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
