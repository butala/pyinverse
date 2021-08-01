import cython
import numpy as np

from pyinverse.volume import read_hyperplanes


def volume_cal2(int m, int d, double [:, :] A, double [:] b):

    cdef double sum_m = 0

#     if(m<2):
#         print("constraint failure!!!")


    # This part detact if this is the base case
    if d==1:

        uplim = float('inf')
        lowlim = -float('inf')

        for i in range(m):
            if(A[i][0]<0):
                #lowlim.append(b[i]/A[i][0])
                if b[i]/A[i][0] > lowlim:
                    lowlim = b[i]/A[i][0]
            elif(A[i][0]>0):
                #uplim.append(b[i]/A[i][0])
                if b[i]/A[i][0] < uplim:
                    uplim = b[i]/A[i][0]
            else:
                # if b[i] > 0:
                #     uplim.append(float('inf'))
                # elif b[i] < 0:
                #     lowlim.append(-float('inf'))
                # else:
                #     print('???????????')
                #     #assert False
                # assert False
                continue
            # if(min(uplim)-max(lowlim)>0):
            #     return min(uplim)-max(lowlim)

        if(uplim-lowlim>0):
            return uplim-lowlim
        else:
            return 0

#         uplim = []
#         lowlim = []
#         for i in range(m):
#             if(A[i][0]<0):
#                 lowlim.append(b[i]/A[i][0])
#             elif(A[i][0]>0):
#                 uplim.append(b[i]/A[i][0])
#             else:
#                 continue
# #         print("basecase:",min(uplim)-max(lowlim))
# #         print("uplim:",uplim)
# #         print("lowlim:",lowlim)
#         if(min(uplim)-max(lowlim)>0):
#             return min(uplim)-max(lowlim)
#         else:
# #             print("no length")
#             return 0
    # if not, the matrix needs to be transformed into lower dimensions
    else:
        #first we need to filter out repeated constraints

        #A_t = A
        #b_t = b

        # A_t = np.empty_like(A)
        # for i in range(A.shape[0]):
        #     for j in range(A.shape[1]):
        #         A_t[i, j] = A[i, j]

        # b_t = np.empty_like(b)
        # for j in range(A.shape[1]):
        #     b_t[j] = b[j]

        # print('start of else')
        # print('A')
        # print(np.array(A))
        # print('b')
        # print(np.array(b))

        A_math = np.zeros((m,d))
        b_math = np.zeros(m)
        m_count = 0


        for i in range(m):
            for j in range(d):
                # print(i, j, A[i, j])
                if A[i, j]!=0:
                    #A_t[i, :] = A[i, :]/abs(A[i][j])
                    for k in range(A.shape[1]):
                        A[i, k] = A[i, k]/abs(A[i, j])
                    b[i] = b[i]/abs(A[i, j])
                    break


        #print(np.array(A))
        # print('first update')

        # print(np.array(a))
        # print()
        # print(np.array(b))
        # print()

        for i in range(m):
            A_me = np.empty_like(A)
            for i2 in range(m):
                for j in range(d):
                    A_me[i2, j] = A[i2, j] - A[i, j]

#             print("A_me:",A_me)
            exist_smaller = False
            # print(np.array(A_me))
#             cross_area_none = 0
            #b_now = b_t[i]

            for c in range(m):
                # A_temp = A[c]+A[i]
                A_temp = np.zeros(d)
                for j in range(d):
                    A_temp[j] = A[c, j] - A[i, j]

                # print('======', min(A_temp), max(A_temp), min(A[c]), max(A[c]), min(A[i]), max(A[i]))


                # print(f'~~~ ({m_count}) {1 if min(A_temp)==0 else 0} {1 if max(A_temp)==0 else 0} {1 if -b[c]>b[i] else 0} {1 if min(A[c])!=0 or max(A[c])!=0 else 0} {1 if min(A[i])!=0 or max(A[i])!=0 else 0}')

                if min(A_temp)==0 and max(A_temp)==0 and -b[c]>b[i] and (min(A[c])!=0 or max(A[c])!=0) and (min(A[i])!=0 or max(A[i])!=0):
#                     cross_area_none = 1
                    # print('???????????????????')
                    return 0


                if min(A_me[c])==0 and max(A_me[c])==0 and (b[c]<b[i] or (b[c]==b[i] and c<i)):
                    exist_smaller = True
                    break



            if not exist_smaller:
                # print(f'--- {m_count}')
                # print(m_count, i)
                A_math[m_count] = A[i]
                b_math[m_count] = b[i]
                m_count = m_count+1

#             if cross_area_none!=0:
#                 return 0


        #here on we can use A_math and b_math to calculate as before
        m_new = m_count
        d_new = d
#         print("A_math:",A_math)
#         print("b_math:",b_math)


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

#                 print("fix_aij:",fix_aij)
#                 print("fix_bi:",fix_bi)
#                 print("i_line:",i_line)
                # print('A_math')
                # print(np.array(A_math))

                # print('b_math')
                # print(np.array(b_math))

                # print('m_new', m_new)
                # print('j', j)
                # print('fix_aij', fix_aij)
                # print('fix_bi', fix_bi)

                # transform into lower dimension
                cal_A = np.zeros((m_new,d_new))
                cal_b = np.zeros(m_new)

                for row in range(m_new):
                    mult = A_math[row][j]/fix_aij
                    #print(mult, A_math[row][j], j)
                    cal_A[row] = A_math[row]-i_line*mult
                    cal_b[row] = b_math[row]-fix_bi*mult

#                 print("A:",A)
#                 print("cal_A:",cal_A)
#                 print("cal_b:",cal_b)


                temp_A0 = np.delete(cal_A,i,axis=0)
                temp_A = np.delete(temp_A0,j,axis=1)
                temp_b = np.delete(cal_b,i,axis=0)



                # print(m_new, d_new)
                # print(cal_A.shape)
                # print(temp_A.shape)

                temp_b2 = np.empty(m_new - 1)
                temp_A2 = np.empty((m_new - 1, d_new - 1))
                temp_A2_i = 0
                for row in range(m_new):
                    if row == i: continue
                    mult = A_math[row][j] / fix_aij
                    temp_b2[temp_A2_i] = b_math[row] - fix_bi * mult
                    # print('}}', temp_A2_i, temp_b2[temp_A2_i], b_math[row], fix_bi, mult)
                    temp_A2_j = 0
                    for col in range(d):
                        if col == j: continue
                        temp_A2[temp_A2_i, temp_A2_j] = A_math[row, col] - A_math[i, col] * mult
                        temp_A2_j += 1
                    temp_A2_i += 1


                # print('1')
                # print(temp_A)

                # print('2')
                # print(m_new, d_new, temp_A2.shape)
                # print(temp_A2)

                # print('allclose?', np.allclose(temp_A, temp_A2))
                # assert np.allclose(temp_A, temp_A2)

                # print('1')
                # print(temp_b)

                # print('2')
                # print(temp_b2)

                # print('allclose?', np.allclose(temp_b, temp_b2))

                # if not np.allclose(temp_A, temp_A2):
                #     row = 0
                #     mult = A_math[row][j]/fix_aij
                #     Z = A_math[row]-i_line*mult
                #     print('mult')
                #     print(mult)
                #     print(Z)
                #     print('A_math[row]')
                #     print(A_math[row])
                #     print('A_math[i]')
                #     print(i_line)
                #     print()

                #     temp_A2 = np.empty((m_new - 1, d_new - 1))
                #     temp_A2_i = 0
                #     for row in range(m_new):
                #         if row == i: continue
                #         mult = A_math[row][j] / fix_aij
                #         print('mult')
                #         print(mult)
                #         print('A_math[row]')
                #         print(A_math[row, :])
                #         print('A_math[i]')
                #         print(np.array(A_math[i, :]))
                #         temp_A2_j = 0
                #         for col in range(d):
                #             if col == j: continue
                #             temp_A2[temp_A2_i, temp_A2_j] = A_math[row, col] - A[i, col] * mult
                #             #print(A_math[row, col], A[i, col])
                #             temp_A2_j += 1
                #         temp_A2_i += 1
                #         break

                #     print(i,j)
                #     print()
                #     print(cal_A)
                #     print()
                #     print(temp_A)
                #     print()
                #     print(temp_A2)
                #     assert False

                #assert False

#                 print("temp_A:",temp_A)
#                 print("temp_b:",temp_b)

                # print('recursion')
                # print(fix_bi, d_new, abs(fix_aij))
                # print(m_new-1, d_new-1)
                # print()
                # print(temp_A)
                # print()
                # print(temp_b)
                #assert False

                sum_m = sum_m+(fix_bi*volume_cal2(m_new-1,d_new-1,temp_A,temp_b)/d_new)/abs(fix_aij)
                # print('ooooooooooooo', sum_m)
#                 print("sum:",sum_m)
        return sum_m




def volume_cal3(int m,
                int d,
                double [:,:] A,
                double [:] b):
    # cdef double sum_m = 0

    # cdef double [:, :] A_t = A
    # cdef double [:] b_t = b

    # cdef double [:, :] A_math = np.zeros((m,d))
    # cdef double [:] b_math = np.zeros(m)
    # cdef int m_count = 0

    # cdef double [:, :] A_me

    # cdef double [:] A_temp

    # cdef int exist_smaller, m_new, d_new

    # cdef double fix_aij, fix_bi
    # cdef double [:] i_line

    # cdef double [:, :] cal_A = A
    # cdef double [:] cal_b = b

    sum_m = 0

    A_me = np.zeros_like(A)
    A_temp = np.zeros(A.shape[1])

    print('here')

    if d == 1:
        print('!!!')
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                print(A[i, j])
        #assert False

        uplim = []
        lowlim = []

        uplim = float('inf')
        lowlim = -float('inf')

        for i in range(m):
            if(A[i][0]<0):
                #lowlim.append(b[i]/A[i][0])
                if b[i]/A[i][0] > lowlim:
                    lowlim = b[i]/A[i][0]
            elif(A[i][0]>0):
                #uplim.append(b[i]/A[i][0])
                if b[i]/A[i][0] < uplim:
                    uplim = b[i]/A[i][0]
            else:
                # if b[i] > 0:
                #     uplim.append(float('inf'))
                # elif b[i] < 0:
                #     lowlim.append(-float('inf'))
                # else:
                #     print('???????????')
                #     #assert False
                assert False
                continue
        # if(min(uplim)-max(lowlim)>0):
        #     return min(uplim)-max(lowlim)
        if(uplim-lowlim>0):
            return uplim-lowlim
        else:
            return 0
    else:
        print('&&&')

        A_t = A
        b_t = b

        A_math = np.zeros((m,d))
        b_math = np.zeros(m)
        m_count = 0

        for i in range(m):
            for j in range(d):
                if A[i][j] != 0:
                    #A_t[i, :] = A[i, :]/abs(A[i][j])
                    for k in range(A_t.shape[1]):
                        A_t[i, k] = A[i, k]/abs(A[i][j])
                    b_t[i] = b[i]/abs(A[i][j])
                    break

        print('boo')
        print(A_t)


        for i in range(m):
            print('boo0')

            # A_me = A_t-A_t[i]
            for i2 in range(A_me.shape[0]):
                for j2 in range(A_me.shape[1]):
                    A_me[i2, j2] = A_t[i2, j2] - A_t[i, j2]

            exist_smaller = False

            print('boo1')

            for c in range(m):
                # A_temp = A_t[c]+A_t[i]
                for j in range(A_t.shape[1]):
                    #print('1')
                    A_temp[j] = A_t[c,j]+A_t[i,j]
                    #print('2')
                if min(A_temp)==0 and max(A_temp)==0 and b_t[c]*-1>b_t[i] and (min(A_t[c])!=0 or max(A_t[c])!=0) and (min(A_t[i])!=0 or max(A_t[i])!=0):
                    return 0

                if min(A_me[c])==0 and max(A_me[c])==0 and (b_t[c]<b_t[i] or (b_t[c]==b_t[i] and c<i)):
                    exist_smaller = True
                    break

            if not exist_smaller:
                print(1)
                print(m_count, i)
                A_math[m_count] = A_t[i]
                b_math[m_count] = b_t[i]
                print(2)
                m_count = m_count+1

        print('boo2')

        m_new = m_count
        d_new = d

        for i in range(m_new):
            if b_math[i] == 0:
                continue
            else:
                for j in range(d_new):
                    if A_math[i][j] != 0:
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
                    print('a')
                    mult = A_math[row][j]/fix_aij
                    print('b')
                    #cal_A[row] = A_math[row]-i_line*mult
                    #cal_b[row] = b_math[row]-fix_bi*mult
                    for j in range(A_math.shape[1]):
                        cal_A[row, j] = A_math[row, j]-i_line[j]*mult
                    cal_b[row] = b_math[row]-fix_bi*mult
                    print('d')

                # temp_A0 = np.delete(cal_A,i,axis=0)
                # temp_A = np.delete(temp_A0,j,axis=1)
                # temp_b = np.delete(cal_b,i,axis=0)

                temp_A = np.zeros_like(cal_A[:-1, :-1])
                temp_A_i = 0
                for i2 in range(cal_A.shape[0]):
                    if i2 == i:
                        continue
                    temp_A_j = 0
                    for j2 in range(cal_A.shape[1]):
                        if j2 == j:
                            continue
                        temp_A[temp_A_i, temp_A_j] = cal_A[i2, j2]
                        temp_A_j += 1
                    temp_A_i += 1

                temp_b = np.zeros_like(cal_b[:-1])
                temp_b_j = 0
                for j2 in range(len(cal_b)):
                    if j2 == j:
                        continue
                    temp_b[temp_b_j] = cal_b[j2]
                    temp_b_j += 1

                sum_m = sum_m+(fix_bi*volume_cal2(m_new-1,d_new-1,temp_A,temp_b)/d_new)/abs(fix_aij)


        return sum_m


if __name__ == '__main__':
    filename = "/Users/butala/src/pyinverse/notebooks/polytope_db/cube/cube_5.ine"
    m, d, G = read_hyperplanes(filename)
    print("Volume using Lass:",volume_cal2(m, d, G[:,1:d+1], G[:,[0]]))
