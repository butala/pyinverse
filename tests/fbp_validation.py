#!/usr/bin/env python3
"""
Low-dimensional validation of 2-D and 3-D filtered backprojection (FBP)
in the ``pyinverse`` tomography testbed.

Questions addressed
-------------------
1. Does reconstruction fidelity improve as the projection density
   (number of angles) increases, in both 2-D and 3-D?
2. Is the ramp filter used by the 3-D FBP (``pyinverse.fbp3``) correct
   for the 3-D "parallel beam, 2-D detector rotating about z" geometry?

Summary of the answer
---------------------
* 2-D FBP (``pyinverse.fbp``) is correct and improves monotonically with
  the number of angles.
* For the geometry used here (a 2-D detector that rotates about the z
  axis -- the setting of ``phantom3.proj`` / ``backproject3`` with
  theta = 0), the exact inversion formula is *slice by slice* 2-D FBP.
  Writing the 3-D object's Fourier transform in the detector frame,
  f = f_u e1 + f_v e2, the change of variables has Jacobian |f_u|, so the
  correct reconstruction filter is the *1-D* ramp |f_u| applied along the
  detector axis transverse to the rotation axis.  ``fbp3.ramp_filter3``
  used to compute the *2-D* ramp sqrt(f_u^2 + f_v^2) instead; that has now
  been fixed (see doc/fbp3-filter-derivation.org).  The two agree only
  when the object is constant along z, which is why the error hid for
  slice-like phantoms.  This harness keeps testing both, as a
  regression test.
* With the wrong (sqrt) filter the reconstruction has a frequency-
  dependent gain sqrt(fx^2+fy^2+fz^2)/sqrt(fx^2+fy^2) != 1, i.e. a bias
  that is *independent of the projection density*; hence the notorious
  "saturation" -- adding projections does not help.
* With the correct |f_u| filter fidelity improves monotonically with the
  number of angles, and the residual plateau is only the (density
  independent) discretisation floor.
* Decisive check: with |f_u|, ``fbp3_theta0`` reproduces an independent
  slice-by-slice application of the (verified) 2-D FBP to machine
  precision, and the wrong sqrt filter does not.
* Tilted detector (``theta0 != 0``): ``backproject3`` also supports a
  planar 2-D detector held at a fixed polar tilt ``theta0`` from the
  rotation axis.  The Jacobian of the (phi, f_u, f_v) chart is
  ``-f_u cos(theta0)``, so the filter is *still* |f_u| but the inversion
  formula carries an explicit ``cos(theta0)`` -- which ``fbp3_theta0``
  applies.  Omitting it inflates the reconstruction by exactly
  ``sec(theta0)``, a pure amplitude error.

  Unlike the untilted case this geometry is intrinsically *incomplete*: a
  frequency is sampled only when ``rho >= |f_z tan theta0|``, so a double
  cone of half-angle ``|theta0|`` about f_z is never measured and the
  error plateaus instead of converging.  Objects invariant along z live on
  ``f_z = 0`` and remain exact at any tilt, which is what makes the
  amplitude error measurable.

Run:  python3 tests/fbp_validation.py
"""
import os
import sys
import types
import numpy as np

# Make the repo importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# Stub out optional, display-only dependencies so the numerical core imports
# cleanly in a headless environment.
# --------------------------------------------------------------------------
for _name in ("imageio", "vtk"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)

from pyinverse.angle import Angle, AngleRegularAxis          # noqa: E402
from pyinverse.axis import RegularAxis                        # noqa: E402
from pyinverse.grid import RegularGrid                        # noqa: E402
from pyinverse.ellipse import Ellipse                         # noqa: E402
from pyinverse.ellipsoid import Ellipsoid                     # noqa: E402
from pyinverse.phantom import Phantom                         # noqa: E402
from pyinverse.fbp import fbp                                 # noqa: E402
from pyinverse import fbp3 as fbp3_mod                        # noqa: E402
from pyinverse.fbp3 import fbp3_theta0                        # noqa: E402

# Keep a handle on the *shipped* filter so repeated sweeps don't capture a
# previously monkeypatched version.
_LIB_RAMP3 = fbp3_mod.ramp_filter3

# numpy < 2 spells the trapezoid rule np.trapz.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Candidate 3-D ramp filters.  NB: ramp_filter3 is *called* as
#     ramp_filter3(grid_uv_ft.Hz())
# so it receives a grid whose axis centers are already in Hz; no further
# conversion is wanted inside the replacement.
FILTERS = {
    # The historical, incorrect filter: isotropic 2-D ramp.
    "sqrt (2-D ramp, wrong)": lambda g: np.sqrt(g.centers[0]**2
                                                + g.centers[1]**2),
    # The correct filter: 1-D ramp along the transverse detector axis.
    "abs(f_u) (correct)": lambda g: np.abs(g.centers[0]),
    # Whatever pyinverse.fbp3 currently ships (should equal abs(f_u)).
    "library ramp_filter3": _LIB_RAMP3,
}


# --------------------------------------------------------------------------
# Minimal stand-in for pyinverse.axes.RegularAxes3 (avoids the VTK/pyviz3d
# import chain).  Reproduces the conventions used by fbp3/backproject3:
# shape == (Nz, Ny, Nx), centers == meshgrid(z, y, x, indexing='ij').
# --------------------------------------------------------------------------
class Axes3:
    def __init__(self, axis_x, axis_y, axis_z):
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z

    @classmethod
    def linspace(cls, lx, ly, lz):
        return cls(RegularAxis.linspace(*lx),
                   RegularAxis.linspace(*ly),
                   RegularAxis.linspace(*lz))

    @property
    def shape(self):
        return (self.axis_z.N, self.axis_y.N, self.axis_x.N)

    @property
    def centers(self):
        return np.meshgrid(self.axis_z.centers,
                           self.axis_y.centers,
                           self.axis_x.centers,
                           indexing="ij")


# --------------------------------------------------------------------------
# Phantoms (analytic indicator functions and their analytic line integrals)
# --------------------------------------------------------------------------
def ellipsoid_phantom():
    """A small, genuinely 3-D ellipsoid phantom (varies along z)."""
    return [
        #    a     b     c     x0    y0    z0    alpha  beta  gamma  rho
        Ellipsoid(0.60, 0.80, 0.30, 0.00, 0.00, 0.00,
                  Angle(deg=0), Angle(deg=0), Angle(deg=0), 1.0),
        Ellipsoid(0.22, 0.22, 0.22, 0.25, 0.00, 0.18,
                  Angle(deg=0), Angle(deg=0), Angle(deg=0), -0.6),
        Ellipsoid(0.16, 0.16, 0.38, -0.25, 0.00, -0.15,
                  Angle(deg=35), Angle(deg=0), Angle(deg=0), 0.5),
    ]


# Semi-axes of the cylindrical phantom's cross-section.  Also used to build
# the independent 2-D reference in the theta0 != 0 checks, so the two cannot
# drift apart.
CYLINDER_AB = (0.60, 0.80)


def cylinder_phantom():
    """A z-independent (cylindrical) phantom: a single huge-c ellipsoid.

    Its projection is independent of v, so *any* detector filter that
    reduces to |f_u| on the f_v = 0 line reproduces the 2-D answer.  This
    is the control that shows why the sqrt-filter bug stayed hidden.

    It is also the control for the tilted-detector (theta0 != 0) checks:
    a z-independent object lives on f_z = 0, which a tilted rotating
    detector always samples, so its data stay *complete* at any tilt and
    exact reconstruction remains possible.
    """
    return [
        Ellipsoid(CYLINDER_AB[0], CYLINDER_AB[1], 100.0, 0.0, 0.0, 0.0,
                  Angle(deg=0), Angle(deg=0), Angle(deg=0), 1.0),
    ]


def eval_ellipsoids(ellipsoids, axes3):
    c_z, c_y, c_x = axes3.centers
    x, y, z = c_x.ravel(), c_y.ravel(), c_z.ravel()
    V = np.zeros_like(x)
    for e in ellipsoids:
        V += e(x, y, z)
    return V.reshape(axes3.shape)


def proj3(grid_uv, ellipsoids, theta, phi):
    Y = np.zeros(grid_uv.shape)
    for e in ellipsoids:
        e.proj(theta, phi, grid_uv, Y=Y)
    return Y


def rel_l2(recon, truth, mask=None):
    if mask is not None:
        recon, truth = recon[mask], truth[mask]
    return np.linalg.norm(recon - truth) / np.linalg.norm(truth)


# ==========================================================================
# 2-D sweep
# ==========================================================================
def run_2d(Na_list, Nx=128, Nt=256, tlim=2.0, plot=None):
    print("\n" + "=" * 74)
    print("2-D FBP  --  modified Shepp-Logan, analytic sinogram  (pyinverse.fbp)")
    print(f"  image {Nx}x{Nx} on [-1,1]^2, {Nt} projection samples on "
          f"[{-tlim},{tlim}]")
    print("=" * 74)
    print(f"{'Na':>6} {'NRMSE':>12} {'NRMSE(central)':>15} {'max|recon|':>11}")

    p = Phantom(key="modified-shepp-logan")
    grid = RegularGrid(RegularAxis.linspace(-1, 1, Nx),
                       RegularAxis.linspace(-1, 1, Nx))
    axis_t = RegularAxis.linspace(-tlim, tlim, Nt)
    truth = p.raster(grid)

    m = Nx // 8
    cent = np.zeros_like(truth, dtype=bool)
    cent[m:-m, m:-m] = True

    out = {}
    recons = {}
    for Na in Na_list:
        axis_theta = RegularAxis.linspace(0, 180, Na, endpoint=False)
        grid_y = RegularGrid(axis_theta, axis_t)
        y = p.sinogram(grid_y)
        recon = fbp(grid, grid_y, y)
        out[Na] = (rel_l2(recon, truth), rel_l2(recon, truth, mask=cent))
        recons[Na] = recon
        print(f"{Na:>6} {out[Na][0]:>12.5f} {out[Na][1]:>15.5f} "
              f"{np.max(np.abs(recon)):>11.4f}")

    if plot is not None:
        plot["2d"] = (truth, recons, out, grid)
    return out


# ==========================================================================
# 3-D sweep
# ==========================================================================
def run_3d(Na_list, n=32, Nu=64, ulim=2.0, filter_name="abs(f_u) (correct)",
           phantom="ellipsoid", theta_deg=0, plot=None):
    print("\n" + "=" * 74)
    print(f"3-D FBP  --  {phantom} phantom, ramp filter = {filter_name}")
    print(f"  volume {n}^3 on [-1,1]^3, detector {Nu}x{Nu} on "
          f"[{-ulim},{ulim}]^2, detector tilt theta0 = {theta_deg} deg")
    print("=" * 74)
    print(f"{'Nphi':>6} {'NRMSE':>12} {'NRMSE(central)':>15} {'max|recon|':>11}")

    ell = {"ellipsoid": ellipsoid_phantom,
           "cylinder": cylinder_phantom}[phantom]()
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    truth = eval_ellipsoids(ell, axes3)
    theta = Angle(deg=theta_deg)

    m = n // 8
    cent = np.zeros_like(truth, dtype=bool)
    cent[m:-m, m:-m, m:-m] = True

    fbp3_mod.ramp_filter3 = FILTERS[filter_name]

    out, recons = {}, {}
    for Nphi in Na_list:
        phi_axis = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180),
                                             Nphi, endpoint=False)
        sinogram3 = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                     for ph in phi_axis.deg.centers]
        recon = fbp3_theta0(axes3, grid_uv, phi_axis, sinogram3, theta0=theta)
        out[Nphi] = (rel_l2(recon, truth), rel_l2(recon, truth, mask=cent))
        recons[Nphi] = recon
        print(f"{Nphi:>6} {out[Nphi][0]:>12.5f} {out[Nphi][1]:>15.5f} "
              f"{np.max(np.abs(recon)):>11.4f}")

    if plot is not None:
        tag = "" if theta_deg == 0 else f"-th{theta_deg}"
        plot[f"3d-{phantom}-{filter_name}{tag}"] = (truth, recons, out, axes3)
    return out


# ==========================================================================
# Decisive cross-check.
#
# For this geometry (theta = 0, detector (u, v) with v = z) the projection at
# height v = z is exactly the 2-D Radon transform of the slice f(., ., z).
# Hence the *exact* 3-D inversion is 2-D FBP applied slice by slice.  We
# reconstruct each slice independently with pyinverse.fbp (after interpolating
# the detector rows to v = z) and compare with fbp3.
# ==========================================================================
def sliced_2d_reference(sinogram3, axes3, grid_uv, phi_axis, theta):
    """Reconstruct slice by slice with the verified 2-D FBP."""
    v = grid_uv.axis_y.centers
    dv = v[1] - v[0]
    SIG = np.asarray(sinogram3)                  # (Nphi, Nv, Nu)
    grid_xy = RegularGrid(axes3.axis_x, axes3.axis_y)
    grid_y = RegularGrid(RegularAxis.linspace(0, 180, phi_axis.N,
                                              endpoint=False), grid_uv.axis_x)
    ref = np.zeros(axes3.shape)
    for k, z in enumerate(axes3.axis_z.centers):
        idx = (z - v[0]) / dv
        j0 = int(np.clip(np.floor(idx), 0, len(v) - 2))
        w = float(np.clip(idx - j0, 0.0, 1.0))
        # projection at height v = z, shape (Nu, Nphi)
        S_k = ((1.0 - w) * SIG[:, j0, :] + w * SIG[:, j0 + 1, :]).T
        ref[k] = fbp(grid_xy, grid_y, S_k)
    return ref


def check_library_filter(Nu=64, ulim=2.0):
    """Confirm the shipped ramp_filter3 is the correct 1-D ramp |f_u|."""
    print("\n" + "=" * 74)
    print("Library filter check:  pyinverse.fbp3.ramp_filter3")
    print("=" * 74)
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    grid_uv_ft, _ = grid_uv.spectrum(np.zeros(grid_uv.shape))
    W = _LIB_RAMP3(grid_uv_ft.Hz())
    ref = np.abs(grid_uv_ft.Hz().centers[0])
    bad = np.sqrt(grid_uv_ft.Hz().centers[0]**2 + grid_uv_ft.Hz().centers[1]**2)
    print(f"  == |f_u|            : {np.array_equal(W, ref)}")
    print(f"  == sqrt(f_u^2+f_v^2) : {np.array_equal(W, bad)}")
    assert np.array_equal(W, ref), "ramp_filter3 is not the 1-D ramp |f_u|"
    print("  OK")


def check_fbp3_equals_sliced_2d(n=24, Nu=48, ulim=2.0, Nphi=64,
                                phantom="ellipsoid"):
    print("\n" + "=" * 74)
    print(f"Cross-check ({phantom}): fbp3 vs independent slice-by-slice 2-D FBP")
    print("=" * 74)
    ell = {"ellipsoid": ellipsoid_phantom,
           "cylinder": cylinder_phantom}[phantom]()
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    theta = Angle(deg=0)
    phi_axis = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180),
                                         Nphi, endpoint=False)
    sinogram3 = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                 for ph in phi_axis.deg.centers]
    truth = eval_ellipsoids(ell, axes3)
    ref = sliced_2d_reference(sinogram3, axes3, grid_uv, phi_axis, theta)

    res = {}
    for name in ("sqrt (2-D ramp, wrong)", "abs(f_u) (correct)",
                 "library ramp_filter3"):
        fbp3_mod.ramp_filter3 = FILTERS[name]
        r = fbp3_theta0(axes3, grid_uv, phi_axis, sinogram3, theta0=theta)
        diff = np.linalg.norm(r - ref) / np.linalg.norm(ref)
        res[name] = rel_l2(r, truth)
        print(f"  {name:<22}  NRMSE vs truth = {rel_l2(r, truth):.5f}"
              f"   rel. L2 vs slice-by-slice 2-D FBP = {diff:.3e}")
        # With the correct filter this must be machine precision; it is the
        # theta = 0 regression guard.  The wrong filter only has to fail for
        # a genuinely 3-D object: for the z-invariant control it is supposed
        # to agree (that is why the bug hid for so long).
        if name != "sqrt (2-D ramp, wrong)":
            assert diff < 1e-10, f"{name} does not reproduce slice-by-slice 2-D FBP"
        elif phantom != "cylinder":
            assert diff > 1e-3, "the wrong filter unexpectedly reproduces 2-D FBP"
    fbp3_mod.ramp_filter3 = _LIB_RAMP3
    return ref, res


# ==========================================================================
# theta0 != 0: the tilted detector
#
# The detector is a planar 2-D array held at a fixed polar tilt theta0 from
# the rotation (z) axis while it rotates in phi.  backproject3 maps a voxel
# p to (u, v) = (e1 . p, e2 . p) with
#
#     e1 = (cos phi, sin phi, 0)
#     e2 = (sin phi sin theta, -cos phi sin theta, cos theta)
#
# so at theta = 0, e2 = (0, 0, 1) and v = z: the detector rows are object
# slices and the problem decouples slice by slice.  At theta != 0 the rows
# are oblique and mix z -- and that is exactly what this section exercises.
#
# Two facts drive the checks below.
#
#  * The Jacobian of the (phi, f_u, f_v) chart is -f_u cos(theta0).  The
#    filter therefore stays |f_u|, but the inversion formula carries an
#    explicit cos(theta0).  fbp3_theta0 applies it.  Dropping it inflates
#    the reconstruction by exactly sec(theta0), uniformly in space.
#
#  * A frequency xi is sampled iff xi . n(phi) = 0 for some phi, with
#    n(phi) = (sin phi cos theta, -cos phi cos theta, -sin theta).  This
#    needs rho >= |f_z tan theta|, so the double cone of half-angle |theta|
#    about f_z is NEVER sampled: for a genuinely 3-D object the data are
#    intrinsically incomplete and no number of projections can fix it.
#    Objects invariant along z live on f_z = 0 and stay complete.
#
# check_theta0_amplitude and check_theta0_vs_2d_fbp are the regression guards
# for the cos(theta0) factor (they fail if it is dropped).  The ray-integral,
# missing-cone and continuity checks characterise the geometry itself and are
# independent of that factor.
# ==========================================================================
def _detector_basis(theta_deg, phi_deg):
    """(e1, e2) exactly as backproject3 builds them."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    e1 = np.array([np.cos(ph), np.sin(ph), 0.0])
    e2 = np.array([np.sin(ph) * np.sin(th), -np.cos(ph) * np.sin(th),
                   np.cos(th)])
    return e1, e2


def numeric_ray_integral(ellipsoids, theta_deg, phi_deg, u, v, N=40001,
                         smax=3.0):
    """Brute-force line integral of *ellipsoids* along the detector ray.

    This is deliberately independent of ``Ellipsoid.proj``: the ray is
    rebuilt from the detector basis (e1, e2) and the phantom is integrated
    along it, so it tests the *convention* the forward projector and the
    backprojector share.
    """
    e1, e2 = _detector_basis(theta_deg, phi_deg)
    e = np.cross(e1, e2)
    s = np.linspace(-smax, smax, N)
    px = u * e1[0] + v * e2[0] + s * e[0]
    py = u * e1[1] + v * e2[1] + s * e[1]
    pz = u * e1[2] + v * e2[2] + s * e[2]
    val = np.zeros(N)
    for el in ellipsoids:
        val += el(px, py, pz)
    return _trapz(val, s)


def missing_cone_energy_fraction(truth, theta_deg):
    """Fraction of the object's (mean-removed) energy in the double cone of
    half-angle *theta_deg* about f_z -- the part of frequency space a tilted
    rotating detector never samples.
    """
    f = truth - truth.mean()
    P = np.abs(np.fft.fftn(f))**2
    nz, ny, nx = truth.shape
    KZ, KY, KX = np.meshgrid(np.fft.fftfreq(nz), np.fft.fftfreq(ny),
                             np.fft.fftfreq(nx), indexing="ij")
    ang = np.arctan2(np.sqrt(KX**2 + KY**2), np.abs(KZ))
    if theta_deg <= 0:
        return 0.0
    return P[ang < np.radians(theta_deg)].sum() / P.sum()


def check_theta0_forward_model(n=24, Nu=48, ulim=2.0, thetas=(0, 15, 30, 45)):
    """(A1) The tilted forward model on the z-invariant phantom.

    Along a tilted ray the path through a cylinder is longer by exactly
    1/cos(theta), and it is still independent of the detector row v (the
    object is invariant along the direction in which the rows fan out).
    Confirms that ``Ellipsoid.proj`` and ``backproject3`` agree on the tilt
    convention.
    """
    print("\n" + "=" * 74)
    print("Tilted forward model (z-invariant phantom, phi = 0)")
    print("=" * 74)
    print(f"{'theta':>6} {'sec(theta)':>11} {'ratio':>10} "
          f"{'ratio*cos':>10} {'v-spread/max':>13}")
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    ell = cylinder_phantom()
    p0 = proj3(grid_uv, ell, Angle(deg=0), Angle(deg=0))
    keep = np.abs(p0) > 1e-3 * np.abs(p0).max()
    for th in thetas:
        p = proj3(grid_uv, ell, Angle(deg=th), Angle(deg=0))
        ratio = np.median(p[keep] / p0[keep])
        spread = np.max(np.ptp(p, axis=0)) / np.max(np.abs(p))
        print(f"{th:>6} {1 / np.cos(np.radians(th)):>11.6f} {ratio:>10.6f} "
              f"{ratio * np.cos(np.radians(th)):>10.6f} {spread:>13.2e}")
        assert abs(ratio * np.cos(np.radians(th)) - 1) < 1e-2, \
            f"tilted projection is not sec(theta) times the untilted one (theta={th})"
        assert spread < 2e-3, f"tilted projection is not v-independent (theta={th})"
    print("  OK: projection scales as sec(theta), v-independent")


def check_theta0_ray_convention(Nu=48, ulim=2.0, thetas=(15, 30, 45),
                                phis=(0, 37, 90)):
    """(A2) The tilted forward model on a genuinely 3-D phantom, validated
    against a brute-force numerical ray integral.
    """
    print("\n" + "=" * 74)
    print("Tilted forward model vs brute-force ray integral (3-D phantom)")
    print("=" * 74)
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    ell = ellipsoid_phantom()
    iu_u = [Nu // 2, Nu // 3, 2 * Nu // 3]
    iv_u = [Nu // 2, 2 * Nu // 3, Nu // 3]
    worst = 0.0
    for th in thetas:
        for ph in phis:
            Y = proj3(grid_uv, ell, Angle(deg=th), Angle(deg=ph))
            hi = 0.05 * np.abs(Y).max()
            for iu, iv in zip(iu_u, iv_u):
                u = grid_uv.axis_x.centers[iu]
                v = grid_uv.axis_y.centers[iv]
                num = numeric_ray_integral(ell, th, ph, u, v)
                ana = Y[iv, iu]
                if max(abs(ana), abs(num)) < hi:
                    continue
                err = abs(ana - num) / max(abs(num), hi)
                worst = max(worst, err)
    print(f"  worst relative mismatch over theta in {thetas}, phi in {phis}: "
          f"{worst:.2e}")
    assert worst < 1e-3, "analytic tilt projection disagrees with ray integral"
    print("  OK: analytic and numerical projections agree")


def check_theta0_amplitude(n=24, Nu=48, ulim=2.0, Nphi=64,
                           thetas=(0, 15, 30, 45)):
    """(B) The cos(theta0) factor.

    On the z-invariant phantom the data are complete at every tilt, so the
    reconstruction must be *theta-independent*.  The pre-fix code returned
    exactly sec(theta) times the correct answer (the counterfactual column
    below), which is a pure amplitude error and therefore invisible to any
    phi-only convergence study.
    """
    print("\n" + "=" * 74)
    print("Reconstruction amplitude vs detector tilt (z-invariant phantom)")
    print("=" * 74)
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    phi_axis = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180), Nphi,
                                         endpoint=False)
    ell = cylinder_phantom()
    truth = eval_ellipsoids(ell, axes3)
    print(f"{'theta':>6} {'rel.L2':>10} {'alpha':>10} {'max|recon|':>11} "
          f"{'|sec-1| pref':>12} {'max|recon| pref':>16}")
    alphas, errs = [], []
    for th in thetas:
        theta = Angle(deg=th)
        sino = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                for ph in phi_axis.deg.centers]
        r = fbp3_theta0(axes3, grid_uv, phi_axis, sino, theta0=theta)
        alpha = np.sum(r * truth) / np.sum(truth * truth)
        rel = rel_l2(r, truth)
        # exactly the pre-fix behaviour, which omitted the cos(theta0):
        r_pre = r / np.cos(theta.rad)
        alphas.append(alpha)
        errs.append(rel)
        print(f"{th:>6} {rel:>10.5f} {alpha:>10.5f} {np.abs(r).max():>11.5f} "
              f"{abs(1 / np.cos(theta.rad) - 1):>12.3f} "
              f"{np.abs(r_pre).max():>16.5f}")
    spread = (max(alphas) - min(alphas)) / np.mean(alphas)
    print(f"  alpha spread over theta = {spread:.2e}; "
          f"rel.L2 spread = {max(errs) - min(errs):.2e}")
    assert spread < 1e-2, "reconstruction amplitude still varies with theta"
    assert max(errs) - min(errs) < 1e-2, "reconstruction error varies with theta"
    print("  OK: cos(theta0) factor present -- amplitude is theta-independent")


def check_theta0_vs_2d_fbp(n=24, Nu=48, ulim=2.0, Nphi=64,
                           thetas=(0, 15, 30, 45)):
    """(C) Decisive check for the tilted case.

    A z-invariant phantom's tilted projection is sec(theta) times its
    untilted one and still v-independent, so an exact tilted inversion must
    reproduce the *2-D* FBP of the cross-section at every tilt.  The
    reference here comes from the analytic 2-D Radon transform of a 2-D
    ``Ellipse`` -- no 3-D code and no cos(theta) involved -- so agreement is
    impossible to fake.  The residual (~1e-3) is the forward model's own
    v-independence error, not a reconstruction error.
    """
    print("\n" + "=" * 74)
    print("Tilted 3-D recon of a z-invariant phantom vs independent 2-D FBP")
    print("=" * 74)
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    phi_axis = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180), Nphi,
                                         endpoint=False)
    ell = cylinder_phantom()

    grid_y = RegularGrid(RegularAxis.linspace(0, 180, Nphi, endpoint=False),
                         grid_uv.axis_x)
    grid_xy = RegularGrid(axes3.axis_x, axes3.axis_y)
    S2d = Ellipse(1.0, CYLINDER_AB[0], CYLINDER_AB[1], 0.0, 0.0,
                  Angle(deg=0)).sinogram(grid_y)
    ref2d = fbp(grid_xy, grid_y, S2d)

    print(f"{'theta':>6} {'rel.L2 vs 2-D FBP':>19} {'z-spread/max':>13} "
          f"{'pre-fix rel.L2':>15} {'pre-fix max|rec|':>17}")
    for th in thetas:
        theta = Angle(deg=th)
        sino = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                for ph in phi_axis.deg.centers]
        r = fbp3_theta0(axes3, grid_uv, phi_axis, sino, theta0=theta)
        ref = np.broadcast_to(ref2d, r.shape)
        d = rel_l2(r, ref)
        dz = np.max(np.ptp(r, axis=0)) / np.max(np.abs(r))
        r_pre = r / np.cos(theta.rad)
        print(f"{th:>6} {d:>19.3e} {dz:>13.2e} {rel_l2(r_pre, ref):>15.3f} "
              f"{np.abs(r_pre).max():>17.5f}")
        assert d < 2e-3, f"tilted recon does not match 2-D FBP (theta={th})"
        assert dz < 2e-3, f"tilted recon of a z-invariant object varies with z"
        if th > 0:
            assert rel_l2(r_pre, ref) > 0.03, \
                "counterfactual does not show the sec(theta) overshoot"
    print("  OK: exact at every tilt; the omitted cos(theta0) would inflate "
          "it by sec(theta0)")


def check_theta0_missing_cone(n=24, Nu=48, ulim=2.0,
                              Nphi_list=(16, 32, 64, 128),
                              thetas=(0, 15, 30, 45)):
    """(D) The missing cone: a genuinely 3-D object at tilt.

    Increasing the number of projections must NOT remove the error once the
    plateau is reached, and the plateau must grow with theta, tracking the
    energy fraction of the never-sampled double cone.
    """
    print("\n" + "=" * 74)
    print("Missing cone: 3-D ellipsoid reconstruction vs detector tilt")
    print("=" * 74)
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    ell = ellipsoid_phantom()
    truth = eval_ellipsoids(ell, axes3)
    print("  " + f"{'theta':>6}" + "".join(f"{'Nphi=' + str(k):>11}"
                                           for k in Nphi_list)
          + f"{'plateau':>11}{'cone frac':>11}{'sqrt(frac)':>12}")
    plateau = {}
    for th in thetas:
        theta = Angle(deg=th)
        row = []
        for Np in Nphi_list:
            pa = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180), Np,
                                           endpoint=False)
            sino = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                    for ph in pa.deg.centers]
            r = fbp3_theta0(axes3, grid_uv, pa, sino, theta0=theta)
            row.append(rel_l2(r, truth))
        frac = missing_cone_energy_fraction(truth, th)
        plateau[th] = row[-1]
        print("  " + f"{th:>6}" + "".join(f"{v:>11.5f}" for v in row)
              + f"{row[-1] - row[-2]:>11.1e}{frac:>11.5f}"
              f"{np.sqrt(frac):>12.4f}")
        assert abs(row[-1] - row[-2]) < 1e-3, \
            f"error is still decreasing at theta={th} -- not a plateau"
        if frac > 1e-3:
            # The unmeasured cone energy shows up (to within a small factor)
            # as the reconstruction error: a quantitative account of the
            # plateau.
            assert 0.5 * np.sqrt(frac) < row[-1] < 2.0 * np.sqrt(frac), \
                f"plateau does not track the missing-cone energy at theta={th}"
    vals = [plateau[th] for th in thetas]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), \
        "error does not increase with detector tilt"
    print("  OK: error plateaus with more projections, grows with theta, and "
          "tracks the missing-cone energy fraction")
    return thetas, plateau, Nphi_list


def check_theta0_continuity(n=24, Nu=48, ulim=2.0, Nphi=64,
                            thetas_deg=(0.001, 0.1, 1.0)):
    """(E) Continuity at theta -> 0, and the theta = 0 regression."""
    print("\n" + "=" * 74)
    print("Continuity of the tilted reconstruction as theta -> 0")
    print("=" * 74)
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    phi_axis = AngleRegularAxis.linspace(Angle(deg=0), Angle(deg=180), Nphi,
                                         endpoint=False)
    ell = ellipsoid_phantom()
    sino0 = [proj3(grid_uv, ell, Angle(deg=0), Angle(deg=ph))
             for ph in phi_axis.deg.centers]
    r0 = fbp3_theta0(axes3, grid_uv, phi_axis, sino0, theta0=Angle(deg=0))
    rates = []
    for th in thetas_deg:
        theta = Angle(deg=th)
        sino = [proj3(grid_uv, ell, theta, Angle(deg=ph))
                for ph in phi_axis.deg.centers]
        r = fbp3_theta0(axes3, grid_uv, phi_axis, sino, theta0=theta)
        d = rel_l2(r, r0)
        rates.append(d / th)
        print(f"  theta = {th:>6g} deg   rel.L2(r_theta, r_0) = {d:.3e}"
              f"   rel.L2/theta = {d / th:.4f} per deg")
    assert max(rates) < 0.1, "tilted recon does not approach the untilted one"
    assert max(rates) / min(rates) < 2.0, \
        "the approach to theta = 0 is not linear in theta"
    print("  OK: the tilt enters linearly and vanishes as theta -> 0")


# ==========================================================================
# Optional figures
# ==========================================================================
def _plot_all(plot):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                    # pragma: no cover
        print(f"\n[matplotlib unavailable ({exc}); skipping figures]")
        return
    import os
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(outdir, exist_ok=True)

    # 2-D fidelity vs projection count
    if "2d" in plot:
        truth, recons, out, grid = plot["2d"]
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        grid.plot(ax[0], truth); ax[0].set_title("2-D phantom")
        for a, Na in zip(ax[1:], sorted(recons)):
            grid.plot(a, recons[Na], vmin=0, vmax=1)
            a.set_title(f"2-D FBP, Na={Na}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "2d_recon.png"), dpi=110)

    # tilted detector: the missing-cone plateau
    if "theta0-plateau" in plot:
        thetas, plateau, Nphi_list = plot["theta0-plateau"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.semilogy(thetas, [plateau[t] for t in thetas], "o-")
        ax.set_xlabel(r"detector tilt $\theta_0$ (deg)")
        ax.set_ylabel(f"rel. L2 error (Nphi = {Nphi_list[-1]})")
        ax.set_title("Missing cone: error grows with tilt,\nand does not fall "
                     "with more projections")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "theta0_plateau.png"), dpi=110)

    # one 3-D case
    for key, val in plot.items():
        if not key.startswith("3d-") or val is None:
            continue
        truth, recons, out, axes3 = val
        ks = sorted(recons)
        k = ks[len(ks) // 2 + 1] if len(ks) > 2 else ks[-1]
        kz = truth.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].imshow(truth[kz], origin="lower", vmin=0, vmax=1)
        ax[0].set_title("3-D phantom, mid z slice")
        ax[1].imshow(recons[k][kz], origin="lower", vmin=0, vmax=1)
        ax[1].set_title(f"{key.split('3d-')[1]}\nNphi={k}")
        fig.tight_layout()
        # Keep generated names shell- and git-friendly: spaces, commas and
        # parentheses in a filter label (e.g. "sqrt (2-D ramp, wrong)") used
        # to leak into the filename.
        name = "".join(c if (c.isalnum() or c in ".-") else "_"
                       for c in key.replace("3d-", "")).strip("_")
        while "__" in name:
            name = name.replace("__", "_")
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=110)
    print(f"\n[figures written to {outdir}]")


# ==========================================================================
if __name__ == "__main__":
    plot = {}

    Na2 = [16, 32, 64, 128, 256, 512]
    run_2d(Na2, plot=plot)

    Na3 = [8, 16, 32, 64, 128, 256]
    run_3d(Na3, n=32, Nu=64, filter_name="sqrt (2-D ramp, wrong)",
           phantom="ellipsoid", plot=plot)
    run_3d(Na3, n=32, Nu=64, filter_name="abs(f_u) (correct)",
           phantom="ellipsoid", plot=plot)

    # Control: a z-independent object is insensitive to the filter choice.
    run_3d([16, 64], n=24, Nu=48, filter_name="sqrt (2-D ramp, wrong)",
           phantom="cylinder")
    run_3d([16, 64], n=24, Nu=48, filter_name="abs(f_u) (correct)",
           phantom="cylinder")

    # Decisive equivalence test against the verified 2-D FBP.
    check_fbp3_equals_sliced_2d(phantom="ellipsoid")
    check_fbp3_equals_sliced_2d(phantom="cylinder")

    # And confirm the shipped filter is the correct one.
    check_library_filter()

    # ---- theta0 != 0: the tilted detector --------------------------------
    check_theta0_forward_model()
    check_theta0_ray_convention()
    check_theta0_amplitude()
    check_theta0_vs_2d_fbp()
    plot["theta0-plateau"] = check_theta0_missing_cone()
    check_theta0_continuity()

    # The same behaviour in the tabular form used above.
    run_3d([16, 64], n=24, Nu=48, filter_name="abs(f_u) (correct)",
           phantom="cylinder", theta_deg=45)
    run_3d([16, 64], n=24, Nu=48, filter_name="abs(f_u) (correct)",
           phantom="ellipsoid", theta_deg=45, plot=plot)

    _plot_all(plot)
    print("\nAll checks passed.")
