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
from pyinverse.ellipsoid import Ellipsoid                     # noqa: E402
from pyinverse.phantom import Phantom                         # noqa: E402
from pyinverse.fbp import fbp                                 # noqa: E402
from pyinverse import fbp3 as fbp3_mod                        # noqa: E402
from pyinverse.fbp3 import fbp3_theta0                        # noqa: E402

# Keep a handle on the *shipped* filter so repeated sweeps don't capture a
# previously monkeypatched version.
_LIB_RAMP3 = fbp3_mod.ramp_filter3

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


def cylinder_phantom():
    """A z-independent (cylindrical) phantom: a single huge-c ellipsoid.

    Its projection is independent of v, so *any* detector filter that
    reduces to |f_u| on the f_v = 0 line reproduces the 2-D answer.  This
    is the control that shows why the sqrt-filter bug stayed hidden.
    """
    return [
        Ellipsoid(0.60, 0.80, 100.0, 0.0, 0.0, 0.0,
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
           phantom="ellipsoid", plot=None):
    print("\n" + "=" * 74)
    print(f"3-D FBP  --  {phantom} phantom, ramp filter = {filter_name}")
    print(f"  volume {n}^3 on [-1,1]^3, detector {Nu}x{Nu} on "
          f"[{-ulim},{ulim}]^2")
    print("=" * 74)
    print(f"{'Nphi':>6} {'NRMSE':>12} {'NRMSE(central)':>15} {'max|recon|':>11}")

    ell = {"ellipsoid": ellipsoid_phantom,
           "cylinder": cylinder_phantom}[phantom]()
    axes3 = Axes3.linspace((-1, 1, n), (-1, 1, n), (-1, 1, n))
    grid_uv = RegularGrid.linspace((-ulim, ulim, Nu), (-ulim, ulim, Nu))
    truth = eval_ellipsoids(ell, axes3)
    theta = Angle(deg=0)

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
        plot[f"3d-{phantom}-{filter_name}"] = (truth, recons, out, axes3)
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
    return ref, res


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

    _plot_all(plot)
