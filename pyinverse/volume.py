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


from fractions import Fraction
from ctypes import c_double, c_size_t

import numpy as np

liblassere = np.ctypeslib.load_library('lassere', '/Users/butala/src/pyinverse/build/lib.macosx-13-x86_64-cpython-311')

lassere_vol = liblassere.lassere_vol
lassere_vol.restype = c_double
lassere_vol.argtypes = [c_size_t,
                        c_size_t,
                        np.ctypeslib.ndpointer(dtype=c_double,
                                               ndim=2,
                                               flags='C'),
                        np.ctypeslib.ndpointer(dtype=c_double,
                                               ndim=1,
                                               flags='C')]

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
                for l in range(N):
                    if l == j:
                        continue
                    A_tilde[k_prime, l_prime] = A[k, l] - A[k, j] * A[i, l] / A[i, j]
                    l_prime += 1

                b_tilde[k_prime] = b[k] - A[k, j] / A[i, j] * b[i]
                k_prime += 1

            vol += b[i] / abs(A[i, j]) * lass_vol_recursion(A_tilde, b_tilde)
        return vol / N

    try:
        return lass_vol_recursion(A, b)
    except EmptyHalfspaceException:
        return 0
    except InfiniteVolumeException:
        return np.inf


def volume_cal(m,d,A,b):
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
            b_now = b_t[i]

            for c in range(m):
                A_temp = A_t[c]+A_t[i]
                if min(A_temp)==0 and max(A_temp)==0 and b_t[c]*-1>b_t[i] and (min(A_t[c])!=0 or max(A_t[c])!=0) and (min(A_t[i])!=0 or max(A_t[i])!=0):
                    return 0
                    break

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
    with open(filename,'rt') as file:  #After code under "with open as" is completed, csvfile is closed
        keywords = file.readlines()
        file.close()

        counter = 0
        G_Hyperplanes = None
        for line in keywords:
            if (counter==3):
                try:
                    a,b,_ = map(str,line.split())
                except:
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
    A = np.array([[ -1,  1],
                  [  2,  1],
                  [1/2, -1],
                  [ -1,  0],
                  [  0, -1]], dtype=float)

    b1 = 1
    b2 = 2
    b3 = 3

    # 0.8333333333333333
    b = np.array([b1, b2, b3, 0, 0], dtype=float)

    print(lass_vol(A, b))

    print(lassere_vol(5, 2, A, b))

    print('-' * 30)

    A = np.array([[-1.        ,  0.        ,  0.        ],
                  [ 1.        ,  0.        ,  0.        ],
                  [ 0.        , -1.        ,  0.        ],
                  [ 0.        ,  1.        ,  0.        ],
                  [ 0.        ,  0.        , -1.        ],
                  [ 0.        ,  0.        ,  1.        ],
                  [-0.92387953,  0.38268343,  0.        ],
                  [ 0.92387953, -0.38268343,  0.        ],
                  [-0.33141357, -0.80010315, -0.5       ],
                  [ 0.33141357,  0.80010315,  0.5       ]])

    b = np.array([ 4.00000000e-01,  2.22044605e-16,  2.85714286e-01,  0.00000000e+00,
                  -7.50000000e-01,  1.25000000e+00,  8.00000000e-01, -4.00000000e-01,
                  -2.50000000e-01,  7.50000000e-01])

    # 0
    print(lass_vol(A, b))
    print(volume_cal(10, 3, A, b))
    print('-' * 30)

    A = np.array([[1, 1]])
    b = np.array([1])

    # inf
    print(lass_vol(A, b))
    #print(volume_cal(1, 2, A, b))

    print('-' * 30)

    # inf
    A = np.array([[-0.13695936, -1.5532242],
                  [ 1.11813139, -0.64901562]])

    b = np.array([-0.50402157,  0.71513002])

    print(lass_vol(A, b))
    # print(volume_cal(2, 2, A, b))
