from linalg import *
from matrix import Matrix
from matrix import EPSILON


# 2차 선형 점화관계식을 풉니다.
def sol_linear():
    a = Matrix.from_input(2, 2)
    x = Matrix.from_input(2, 1)

    d = a.get_element(0, 0)*a.get_element(0, 0) + 4*a.get_element(0, 1)
    if d > 0:
        lam1 = (a.get_element(0, 0) + sqrt(d))/2.0
        lam2 = (a.get_element(0, 0) - sqrt(d))/2.0

        b = Matrix(2, 2)
        b.set_element(0, 0, lam1)
        b.set_element(0, 1, lam2)
        b.set_element(1, 0, 1)
        b.set_element(1, 1, 1)

        c = inverse_matrix(b)*x
        print(f'xn : {c.get_element(0, 0)}*({lam1})^n + {c.get_element(1, 0)}*({lam2})^n')
    elif d == 0:
        lam = a.get_element(0, 0)/2.0
        b = Matrix(2, 2)
        b.set_element(0, 0, lam)
        b.set_element(0, 1, lam)
        b.set_element(1, 0, 1)

        c = inverse_matrix(b)*x
        print(f'xn : {c.get_element(0, 0)}*({lam})^n + {c.get_element(1, 0)}*n*({lam})^n')

    else:
        print("no solution")


# 2 차 선형 미분방정식을 푼다
def sol_linear_diff():
    a = Matrix.from_input()
    d, p = diagonalization(a)
    x0 = Matrix.from_input(a.col_size, 1)
    c = inverse_matrix(p)*x0
    # print(c)
    # print(d)
    # print(p)
    for i in range(a.col_size):
        tmp = ""
        for j in range(a.col_size):
            tmp += "{0}e^{1}t ".format(c.get_element(j, 0)*p.get_element(i, j), d.get_element(j, j))
            if j+1 < a.col_size:
                tmp += '+ '
        print(tmp)


def sol_linear_diff_b():
    a = Matrix.from_input()
    d, p = diagonalization(a)
    b = Matrix.from_input(a.col_size, 1)
    b1 = inverse_matrix(a)*b
    x0 = Matrix.from_input(a.col_size, 1) + b1
    c = inverse_matrix(p) * x0
    for i in range(a.col_size):
        tmp = ""
        for j in range(a.col_size):
            tmp += "{0}e^{1}t + {2} ".format(c.get_element(j, 0)*p.get_element(i, j), d.get_element(j, j),
                                             b1.get_element(i, 0))
            if j+1 < a.col_size:
                tmp += '+ '
        print(tmp)


# sol_linear()
a = Matrix.from_input()
d, p = diagonalization(a)
print(inverse_matrix(p))
print(d)
print(p)
