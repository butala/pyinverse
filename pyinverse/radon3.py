import numpy as np
import scipy as sp
import vtk


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
