from dataclasses import dataclass

import numpy as np
import scipy
import vtk

from pyinverse.angle import Angle


"""
Take a look here:

X-ray transform and 3D Radon transform for ellipsoids and tetrahedra
"""


def ellipsoid_proj(ellipsoid, theta, phi, grid, Y=None):
    """
    Return the X-ray transform of *ellipsoid* at the azimuthal
    angle *phi* (ranging from -pi to pi, though negative angles are
    redundant due to symmetry) and polar angle *theta* (ranging from
    -pi/2 to pi/2). If *deg* is `True` then these angles are given in
    degrees and in radians otherwise. The result is stored in *Y* if
    it is provided and in the new, appropriately sized array
    otherwise. The U and V coordinates in the projection plan are
    specified by the x axis and y axis of *grid*, respectively.

    The angular conventions follow Fessler's notes
    (http://web.eecs.umich.edu/~fessler/book/c-tomo-prop.pdf).

    In the table below, the unit vector e is parallel to the line
    integrals (so the sign does not matter from the viewpoint of the
    integration) and normal to the projection plane. The vectors e1
    and e2 specify the projection plane coordinates.

    |-------+------+-----+-----+-----|
    | theta |  phi |   e |  e1 |  e2 |
    |-------+------+-----+-----+-----|
    |     0 |    0 |   y |   x |   z |
    |     0 | pi/2 |  -x |   y |   z |
    |     0 |   pi |  -y |  -x |   z |
    |  pi/2 |    0 |   z |   x |  -y |
    | -pi/2 |    0 |  -z |   x |   y |
    |  pi/2 | pi/2 |   z |   y |   x |
    |-------+------+-----+-----+-----|
    """
    if Y is None:
        Y = np.zeros((grid.shape))
    # Problem 4.15 from Fessler's notes
    e_vec0 = np.array([-phi.sin * theta.cos,
                        phi.cos * theta.cos,
                        theta.sin])

    e1_vec0 = np.array([phi.cos,
                        phi.sin,
                        0])

    e2_vec0 = np.array([ phi.sin * theta.sin,
                        -phi.cos * theta.sin,
                         theta.cos])

    # rotation property
    e_vec = ellipsoid.R_matrix.T @ e_vec0
    e1_vec = ellipsoid.R_matrix.T @ e1_vec0
    e2_vec = ellipsoid.R_matrix.T @ e2_vec0

    U, V = grid.centers

    # shift property
    p0 = np.array([ellipsoid.x0, ellipsoid.y0, ellipsoid.z0])
    U0 = np.dot(e1_vec0, p0)
    V0 = np.dot(e2_vec0, p0)

    U = U.flat - U0
    V = V.flat - V0

    p = e1_vec[:, np.newaxis] @ U[np.newaxis, :] + e2_vec[:, np.newaxis] @ V[np.newaxis, :]

    M = np.diag([1/ellipsoid.a, 1/ellipsoid.b, 1/ellipsoid.c])
    M_e = M @ e_vec

    A = np.dot(M_e, M_e)

    M_p = M @ p

    B = np.einsum('i,ij->j', M_e, M_p)
    C = np.einsum('ij,ij->j', M_p, M_p) - 1

    I = B**2 >= A * C
    Y.flat[I] += 2/A * np.sqrt(B[I]**2 - A*C[I]) * ellipsoid.rho

    return Y


def ellipsoid_ft(ellipsoid, kx, ky, kz, optimize=False):
    """*kx*, *k_y*, and *k_z* are in Hz"""
    # Cheng Guan Koay, Joelle E. Sarlls, and Evren Ozarslan,
    # Three-Dimensional Analytical Magnetic Resonance Imaging Phantom
    # in the Fourier Domain, Magnetic Resonance in Medicine 58:430–436
    # (2007)
    assert kx.shape == ky.shape == kz.shape
    I0 = np.isclose(kx, 0) & np.isclose(ky, 0) & np.isclose(kz, 0)
    R = ellipsoid.R_matrix
    kxyz = np.array([kx, ky, kz])
    kxyz_tilde = np.einsum('ij,jklm->iklm', ellipsoid.R_matrix.T, kxyz, optimize=optimize)
    kx_tilde = kxyz_tilde[0, :, :, :]
    ky_tilde = kxyz_tilde[1, :, :, :]
    kz_tilde = kxyz_tilde[2, :, :, :]
    K = np.sqrt((ellipsoid.a * kx_tilde)**2 + (ellipsoid.b * ky_tilde)**2 + (ellipsoid.c * kz_tilde)**2)
    delta = np.array([ellipsoid.x0, ellipsoid.y0, ellipsoid.z0])
    P = np.exp(-1j * 2 * np.pi * np.einsum('i,ijkl->jkl', np.array(delta), kxyz, optimize=optimize))
    out = np.empty_like(kx, dtype=complex)
    out[I0] = ellipsoid.rho * (4/3) * np.pi * ellipsoid.a * ellipsoid.b * ellipsoid.c
    J = ~I0
    out[J] = ellipsoid.rho * ellipsoid.a * ellipsoid.b * ellipsoid.c * (np.sin(2 * np.pi * K[J]) - 2 * np.pi * K[J] * np.cos(2 * np.pi * K[J])) / (2 * np.pi**2 * K[J]**3)
    out *= P
    return out


@dataclass
class Ellipsoid:
    a: float
    b: float
    c: float

    x0: float
    y0: float
    z0: float

    alpha: Angle
    beta: Angle
    gamma: Angle

    rho: float


    @property
    def R_matrix(self):
        """
        Return the rotation matrix corresponding to the Z-X-Z
        Euler angle parameters alpha, beta, and gamma.
        """
        try:
            return self._R_matrix
        except AttributeError:
            self._R_matrix = scipy.spatial.transform.Rotation.from_euler('ZXZ',
                                                                         [self.alpha.deg, self.beta.deg, self.gamma.deg],
                                                                         degrees=True).as_matrix()
            return self.R_matrix


    def __call__(self, x, y, z, Y=None):
        """
        Evaluate the ellipsoid indicator function for those points
        given in (possibly same vector length) coordinate arguments
        *x*, *y*, and *z*. If the coordinate is interior to the
        ellipsoid then return rho, otherwise return 0. If *Y* is
        given, return the result in *Y* and return a new appropriately
        sized array if *Y* is `None`.
        """
        assert x.ndim == y.ndim == z.ndim == 1
        assert x.shape == y.shape == z.shape
        M = np.diag((1/self.a, 1/self.b, 1/self.c))
        if Y is None:
            Y = np.zeros_like(x)
        else:
            assert Y.shape == x.shape
        p = np.stack((x, y, z)) - np.array([self.x0, self.y0, self.z0])[:, np.newaxis]
        M_p = (M @ self.R_matrix.T) @ p
        I = np.einsum('ij,ij->j', M_p, M_p) <= 1
        Y[I] = self.rho
        return Y


    def actor(self):
        """
        """
        # ellipsoid
        ellipsoid = vtk.vtkParametricEllipsoid()
        ellipsoid.SetXRadius(self.a)
        ellipsoid.SetYRadius(self.b)
        ellipsoid.SetZRadius(self.c)
        # source
        ellipsoid_source = vtk.vtkParametricFunctionSource()
        ellipsoid_source.SetParametricFunction(ellipsoid)
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(ellipsoid_source.GetOutputPort())
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # apply standard (Z-X-Z) Euler angle rotation

        actor.RotateZ(self.alpha.deg)
        actor.RotateX(self.beta.deg)
        actor.RotateZ(self.gamma.deg)
        actor.SetPosition(self.x0, self.y0, self.z0)
        return actor


    def proj(self, theta, phi, grid, Y=None):
        """
        Return the projection of the ellipsoid. See the detailed
        documentation in :func:`ellipsoid_proj`.
        """
        return ellipsoid_proj(self, theta, phi, grid, Y=Y)


    def fourier_transform(self, fx, fy, fz):
        """
        """
        return ellipsoid_ft(self, fx, fy, fz)


if __name__ == '__main__':
    # e = Ellipsoid(0.6900, 0.9200, 0.810, 0, 0, 0, 0, 0, 0, 1.0)
    # e = Ellipsoid(0.6900, 0.9200, 0.810, 0, -0.25, 0, 0, 0, 0, 1.0)
    # e = Ellipsoid(0.6900, 0.9200, 0.810, 0, 0, -0.25, 0, 0, 0, 1.0)
    # e = Ellipsoid(10*0.046, 10*0.023, 10*0.02, 5*-0.08, -0.65, -0.25, 0, 0, 0, 0.1)

    e = Ellipsoid(0.31, 0.11, 0.22, 0.22, 0, -0.25,
                  Angle(deg=72), Angle(deg=0), Angle(deg=0),
                  -0.2)

    from pyviz3d.viz import Renderer

    actor = e.actor()
    actor.GetProperty().SetColor(1, 0, 0)

    ren = Renderer()
    ren.add_actor(actor)
    ren.axes_on((-1, 1, -1, 1, -1, 1))
    ren.reset_camera()

    ren.start()
