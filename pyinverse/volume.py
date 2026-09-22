"""Polytope volume computation.

Two independent implementations of the volume of the polytope
:math:`\\{x \\mid A x \\le b\\}`:

``volume_cal``
    Pure Python/NumPy.  Always available; the reference implementation.
``lasserre_vol``
    A thin wrapper around the optional C extension in ``pyinverse/lasserre``.
    The shared library is loaded *lazily*, on the first call, so that importing
    :mod:`pyinverse` and using the analytic Radon path never requires the
    extension to be built -- see :func:`lasserre_available`.  Set
    ``PYINVERSE_LASSERRE_DIR`` to point at a built library, or
    ``pip install -e .`` (with a C compiler) to build it.
"""

#Given (A,b) as H-form data, and V as a list of vertices
#P={x|Ax<b}, P=conv(V)

#This is the part of code realizing Lasserre's Method in Bueler2000 paper
#Chapter 3.2, Page 10 specifically


#Update on Aug.11th, 2020
#This code is known to be useful for at least following tests, but may have some other bug.
#Will work on better solutions on deleting linearly-dependent constraints

#Updated on Aug.17th, 2020
#Problem of not considering conflicting upbound and lowerbound is now fixed,
#but the efficiency of the code should be adapted further more.

#Updated on Aug.25th, 2020
#Several tests have been used on the code, including cube_8(and lower dimensions),
#cross_6(and lower dimensions), cc_8_6(and lower dimensions), and Fm_4(and lower dimensions).
#Problems with remaining testcases are either the code is too slow
#or it cannot handle fractions for now, still working on the script and other methods.


import ctypes
import importlib.util
import os
import platform
import sys
from ctypes import c_double, c_size_t
from fractions import Fraction
from pathlib import Path

import numpy as np

#: Base name of the optional C extension, e.g. ``lasserre.cpython-312-darwin.so``.
#: This will not work on Windows but may work on Linux.
LASSERRE_LIB_NAME = (
    f"lasserre.{sys.implementation.name}-{sys.version_info.major}"
    f"{sys.version_info.minor}-{platform.system().lower()}.so"
)

_lasserre_vol_c = None


def lasserre_candidates():
    """Yield, in order, the paths searched for the ``lasserre`` shared library.

    The search order is:

    1. ``$PYINVERSE_LASSERRE_DIR``, if set;
    2. the location of an installed top-level extension module named
       ``lasserre`` (what ``pip install`` produces for
       ``Extension(name='lasserre', ...)``);
    3. the repository root and then the package directory (what an in-place
       ``pip install -e .`` or ``python setup.py build_ext --inplace`` produces).
    """
    name = LASSERRE_LIB_NAME
    env_dir = os.environ.get("PYINVERSE_LASSERRE_DIR")
    if env_dir:
        yield Path(env_dir) / name
    try:
        spec = importlib.util.find_spec("lasserre")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        yield Path(spec.origin)
    here = Path(__file__).resolve().parent
    yield here.parent / name
    yield here / name


def _load_lasserre():
    """Load (once) and return the ``lasserre_vol`` entry point of the extension."""
    global _lasserre_vol_c
    if _lasserre_vol_c is None:
        for candidate in lasserre_candidates():
            if not candidate.is_file():
                continue
            lib = ctypes.CDLL(str(candidate))
            fn = lib.lasserre_vol
            fn.restype = c_double
            fn.argtypes = [c_size_t,
                           c_size_t,
                           np.ctypeslib.ndpointer(dtype=c_double,
                                                  ndim=2,
                                                  flags="C"),
                           np.ctypeslib.ndpointer(dtype=c_double,
                                                  ndim=1,
                                                  flags="C")]
            _lasserre_vol_c = fn
            break
        else:
            searched = ", ".join(str(p) for p in lasserre_candidates())
            raise ImportError(
                f"the optional 'lasserre' C extension ({LASSERRE_LIB_NAME}) was "
                f"not found; searched {searched}. It is only needed for the fast "
                "polytope-volume path (``lasserre_vol``); the analytic Radon "
                "transform and ``volume_cal`` do not use it. Build it with "
                "`pip install -e .` (requires a C compiler), or set "
                "PYINVERSE_LASSERRE_DIR to its directory."
            )
    return _lasserre_vol_c


def lasserre_available():
    """Return True if the optional ``lasserre`` C extension can be loaded."""
    try:
        _load_lasserre()
    except ImportError:
        return False
    return True


def lasserre_vol(m, d, A, b):
    """Volume of :math:`\\{x \\mid A x \\le b\\}` via the optional C extension.

    Args:
        m (int): number of half-space constraints (rows of *A*).
        d (int): dimension of the space.
        A (array_like): ``(m, d)`` constraint matrix.
        b (array_like): ``(m,)`` constraint vector.

    Returns:
        float: the volume.

    Raises:
        ImportError: if the extension has not been built.  Use
            :func:`lasserre_available` to test for this, or
            :func:`volume_cal` for the always-available pure Python path.
    """
    fn = _load_lasserre()
    A = np.ascontiguousarray(A, dtype=c_double)
    b = np.ascontiguousarray(b, dtype=c_double)
    return fn(m, d, A, b)

class EmptyHalfspaceException(Exception):
    pass

class InfiniteVolumeException(Exception):
    pass

class AllZeroRow(Exception):
    pass


def first_nonzero_column(A, i):
    """
    """
    M, N = A.shape
    assert i >= 0 and i < M
    for j in range(N):
        if not np.allclose(A[i, j], 0):
            return j
    raise AllZeroRow()


def normalize_constraints(A, b):
    """
    Normalize (scale to set first nonzero column to 1) and remove all 0 rows.
    """
    M, N = A.shape

    A_out = []
    b_out = []

    for i in range(M):
        try:
            j = first_nonzero_column(A, i)
        except AllZeroRow:
            continue
        A_out.append(A[i, :] / abs(A[i, j]))
        b_out.append(b[i] / abs(A[i, j]))
    return np.atleast_2d(np.array(A_out)), np.array(b_out)


def filter_parallel_constraints(A, b):
    """
    """
    M, N = A.shape

    A_out = []
    b_out = []

    parallel_halfspaces = set()

    for i in range(M):
        if i in parallel_halfspaces:
            continue
        smallest_b = None
        try:
            j = first_nonzero_column(A, i)
        except AllZeroRow:
            continue
        scale_factor_i = abs(A[i, j])
        for k in range(i+1, M):
            if np.allclose(A[k, j], 0):
                continue
            scale_factor_k = abs(A[k, j])

            if np.allclose(A[i, :] / scale_factor_i, A[k, :] / scale_factor_k):
                # parallel half spaces detected --- remove the larger one as it is redundant
                parallel_halfspaces.add(k)
                if b[i] <= b[k]:
                    smallest_b = b[i]
                else:
                    smallest_b = b[k]
            elif np.allclose(A[i, :] / scale_factor_i, -A[k, :] / scale_factor_k):
                # parallel half spaces detected
                if -b[k] > b[i]:
                    # the half spaces do not overlap
                    raise EmptyHalfspaceException()
        A_out.append(A[i, :])
        if smallest_b is not None:
            b_out.append(smallest_b)
        else:
            b_out.append(b[i])

    return np.atleast_2d(np.array(A_out)), np.array(b_out)


def lass_vol(A, b):
    """
    """
    def lass_vol_recursion(A, b):
        M, N = A.shape
        assert b.ndim == 1 and len(b) == M

        if M == 1:
            raise InfiniteVolumeException()

        # base case
        if N == 1:
            I_positive = A.flat > 0
            I_negative = A.flat < 0

            if sum(I_positive) == 0 or sum(I_negative) == 0:
                raise InfiniteVolumeException()
            else:
                vol = max(0, np.min(b[I_positive] / A.flat[I_positive]) - np.max(b[I_negative] / A.flat[I_negative]))
            return vol

        A, b = normalize_constraints(A, b)
        A, b = filter_parallel_constraints(A, b)

        M, N = A.shape
        assert b.ndim == 1 and len(b) == M

        vol = 0
        A_tilde = np.empty((M-1, N-1))
        b_tilde = np.empty(M-1)

        for i in range(M):
            if np.allclose(b[i], 0):
                 continue

            try:
                j = first_nonzero_column(A, i)
            except AllZeroRow():
                continue

            k_prime = 0
            for k in range(M):
                if k == i:
                    continue

                l_prime = 0
                for ell in range(N):
                    if ell == j:
                        continue
                    A_tilde[k_prime, l_prime] = A[k, ell] - A[k, j] * A[i, ell] / A[i, j]
                    l_prime += 1

                b_tilde[k_prime] = b[k] - A[k, j] / A[i, j] * b[i]
                k_prime += 1

            try:
                vol += b[i] / abs(A[i, j]) * lass_vol_recursion(A_tilde, b_tilde)
            except EmptyHalfspaceException:
                continue
        assert vol >= 0
        return vol / N

    try:
        return lass_vol_recursion(A, b)
    except EmptyHalfspaceException:
        return 0
    except InfiniteVolumeException:
        return np.inf


def volume_cal(m, d, A, b):
    """Volume of the polytope ``{x | A x <= b}`` (pure Python/NumPy).

    This is a recursive half-space decomposition.  It is the always-available
    reference implementation; :func:`lasserre_vol` is the optional fast path.

    Args:
        m (int): number of half-space constraints (rows of *A*).
        d (int): dimension of the space.
        A (array_like): ``(m, d)`` constraint matrix with ``A x <= b``.
        b (array_like): ``(m,)`` constraint vector.

    Returns:
        float: the volume.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.shape != (m, d):
        raise ValueError(f'A must have shape ({m}, {d}), got {A.shape}')
    if b.shape != (m,):
        raise ValueError(f'b must have shape ({m},), got {b.shape}')
    sum_m = 0


    # This part detact if this is the base case
    if d==1:
        uplim = []
        lowlim = []
        for i in range(m):
            if(A[i][0]<0):
                lowlim.append(b[i]/A[i][0])
            elif(A[i][0]>0):
                uplim.append(b[i]/A[i][0])
            else:
                continue
        if(min(uplim)-max(lowlim)>0):
            return min(uplim)-max(lowlim)
        else:
            return 0
    # if not, the matrix needs to be transformed into lower dimensions
    else:
        #first we need to filter out repeated constraints

        A_t = A/1
        b_t = b/1

        A_math = np.zeros((m,d))
        b_math = np.zeros(m)
        m_count = 0

        for i in range(m):
            for j in range(d):
                if A[i][j]!=0:
                    A_t[i] = A[i]/abs(A[i][j])
                    b_t[i] = b[i]/abs(A[i][j])
                    break

        for i in range(m):
            A_me = A_t-A_t[i]
            exist_smaller = 0
            b_t[i]

            for c in range(m):
                A_temp = A_t[c]+A_t[i]
                if (min(A_temp) == 0 and max(A_temp) == 0 and b_t[c]*-1 > b_t[i]
                        and (min(A_t[c]) != 0 or max(A_t[c]) != 0)
                        and (min(A_t[i]) != 0 or max(A_t[i]) != 0)):
                    return 0

                if min(A_me[c])==0 and max(A_me[c])==0 and (b_t[c]<b_t[i] or (b_t[c]==b_t[i] and c<i)):
                    exist_smaller = 1
                    break

            if exist_smaller!=1:
                A_math[m_count] = A_t[i]
                b_math[m_count] = b_t[i]
                m_count = m_count+1

        #here on we can use A_math and b_math to calculate as before
        m_new = m_count
        d_new = d

        for i in range(m_new):
            if b_math[i]==0:
                continue
            else:
                for j in range(d_new):
                    if A_math[i][j]!=0:
                        break

                fix_aij = A_math[i][j]
                fix_bi = b_math[i]
                i_line = A_math[i]

                if fix_aij==0:
                    continue

                # transform into lower dimension
                cal_A = np.zeros((m_new,d_new))
                cal_b = np.zeros(m_new)

                for row in range(m_new):
                    mult = A_math[row][j]/fix_aij
                    cal_A[row] = A_math[row]-i_line*mult
                    cal_b[row] = b_math[row]-fix_bi*mult

                temp_A0 = np.delete(cal_A,i,axis=0)
                temp_A = np.delete(temp_A0,j,axis=1)
                temp_b = np.delete(cal_b,i,axis=0)

                sum_m = sum_m+(fix_bi*volume_cal(m_new-1,d_new-1,temp_A,temp_b)/d_new)/abs(fix_aij)
        return sum_m


# This code is written to read in .ine files and retrieve corresponding
# m, d, A, and b for the main function above.
# There are still some improvement space with the file-reading function,
# for example, it cannot read in numbers in fraction forms for now,
# and I'm thinking of ways to do that.

# Code latest updated on Aug.26th,2020.

def read_hyperplanes(filename):
    with open(filename) as file:  #After code under "with open as" is completed, csvfile is closed
        keywords = file.readlines()
        file.close()

        counter = 0
        G_Hyperplanes = None
        for line in keywords:
            if (counter==3):
                try:
                    a, b, _ = map(str, line.split())
                except Exception:
                    continue
                G_m = int(a)
                G_d = int(b)-1
                G_Hyperplanes = np.zeros((G_m,G_d+1))

            elif (counter>=4 and counter<4+G_m):
                op = map(str,line.split())
                row = list(op)
                s_c = 0
                for i in row:
                    try:
                        G_Hyperplanes[counter-4][s_c] = float(i)
                    except ValueError:
                        G_Hyperplanes[counter-4][s_c] = float(Fraction(i))
                    s_c = s_c +1

            counter = counter+1
    return [G_m,G_d,G_Hyperplanes]


if __name__ == '__main__':
    # Self-check against known polytope volumes.  Run: python3 -m pyinverse.volume
    cases = [
        ('unit square',
         np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float),
         np.array([0, 1, 0, 1], dtype=float),
         1.0),
        ('unit triangle',
         np.array([[-1, 0], [0, -1], [1, 1]], dtype=float),
         np.array([0, 0, 1], dtype=float),
         0.5),
        ('unit cube',
         np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                   [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=float),
         np.array([0, 1, 0, 1, 0, 1], dtype=float),
         1.0),
        ('unit simplex in 3-D',
         np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]], dtype=float),
         np.array([0, 0, 0, 1], dtype=float),
         1/6),
    ]
    ok = True
    for name, A, b, expected in cases:
        got = volume_cal(*A.shape, A, b)
        agree = np.isclose(got, expected)
        ok = ok and agree
        print(f'{name:22s} volume_cal = {got:.12g}  expected {expected:.12g}  '
              f'{"ok" if agree else "MISMATCH"}')
    print(f'{"lasserre C extension":22s} available: {lasserre_available()}')
    sys.exit(0 if ok else 1)
