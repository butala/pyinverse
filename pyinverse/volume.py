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




import numpy as np


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
                a,b,_ = map(str,line.split())
                G_m = int(a)
                G_d = int(b)-1
                G_Hyperplanes = np.zeros((G_m,G_d+1))

            elif (counter>=4 and counter<4+G_m):
                op = map(str,line.split())
                row = list(op)
                s_c = 0
                for i in row:
                    G_Hyperplanes[counter-4][s_c] = float(i)
                    s_c = s_c +1

            counter = counter+1
    return [G_m,G_d,G_Hyperplanes]
