from matrix import Matrix
from matrix import EPSILON
from math import *


def identify(n: int) -> Matrix:
    return Matrix.identify(n)


def transpose(mat: Matrix) -> Matrix:
    return mat.T()


def zeros(r: int, c: int) -> Matrix:
    return Matrix(r, c)


def ones(r: int, c: int) -> Matrix:
    ret = Matrix(r, c)
    for i in range(r):
        for j in range(c):
            ret.set_element(i, j, 1.)

    return ret


def copy_matrix(m: Matrix) -> Matrix:
    return m.copy()


def dot(col_v1: Matrix, col_v2: Matrix) -> float:
    ret = 0
    for i in range(col_v1.row_size):
        ret += col_v1.get_element(i, 0)*col_v2.get_element(i, 0)
    return ret


def length(col_vec: Matrix) -> float:
    return sqrt(dot(col_vec, col_vec))


def normalize(col_vec: Matrix) -> Matrix:
    return (1/length(col_vec))*col_vec


def power_method(mat: Matrix, x0: Matrix = None, max_iter=100) -> list:
    if x0 is None:
        x0 = ones(mat.row_size, 1)

    x1 = copy_matrix(x0)
    m, y0 = None, None
    for it in range(max_iter):
        y0 = copy_matrix(x1)
        m = 0
        for r in range(y0.row_size):
            if abs(m) < abs(y0.get_element(r, 0)):
                m = y0.get_element(r, 0)
        for r in range(y0.row_size):
            y0.eo2_row_multiple_x(r, 1/m)

        x1 = mat*y0

    return [m, y0]


def is_reducible_matrix(mat: Matrix) -> bool:
    n = mat.col_size
    return (identify(n) + mat)**(n-1) > zeros(mat.row_size, mat.col_size)


def rref(mat: Matrix) -> Matrix:
    a = copy_matrix(mat)

    c = 0
    for r in range(0, a.row_size):
        if c >= a.col_size:
            break

        max_row = r
        max_val = a.get_element(r, c)
        for i in range(r+1, a.row_size):
            if abs(max_val) < abs(a.get_element(i, c)):
                max_row = i
                max_val = a.get_element(i, c)

        if abs(max_val) <= EPSILON:
            c += 1
            continue

        if max_row != r:
            a.eo1_row_swap(r, max_row)

        for i in range(r+1, a.row_size):
            d = a.get_element(i, c)/max_val
            a.eo3_r1_multiple_x_add_r2(r, -d, i)

        a.eo2_row_multiple_x(r, 1/max_val)
        c += 1

    return a


def det(mat: Matrix) -> float:
    a = copy_matrix(mat)
    ret = 1.0
    c = 0
    for r in range(0, a.row_size):
        if c >= a.col_size:
            break

        max_row = r
        max_val = a.get_element(r, c)
        for i in range(r+1, a.row_size):
            if abs(max_val) < abs(a.get_element(i, c)):
                max_row = i
                max_val = a.get_element(i, c)

        if abs(max_val) <= EPSILON:
            return 0

        if max_row != r:
            a.eo1_row_swap(r, max_row)
            ret = -ret

        for i in range(r+1, a.row_size):
            d = a.get_element(i, c)/max_val
            a.eo3_r1_multiple_x_add_r2(r, -d, i)

        ret *= max_val
        c += 1
    return ret


def inverse_matrix(mat: Matrix) -> Matrix:
    if mat.col_size != mat.row_size or abs(det(mat)) <= EPSILON:
        return Matrix(0, 0)

    a = copy_matrix(mat)
    identify_mat = identify(mat.col_size)
    c = 0
    for r in range(0, a.row_size):
        if c >= a.col_size:
            break

        max_row = r
        max_val = a.get_element(r, c)
        for i in range(r+1, a.row_size):
            if abs(max_val) < abs(a.get_element(i, c)):
                max_row = i
                max_val = a.get_element(i, c)

        if abs(max_val) <= EPSILON:
            return Matrix(0, 0)

        if max_row != r:
            a.eo1_row_swap(r, max_row)
            identify_mat.eo1_row_swap(r, max_row)

        for i in range(0, a.row_size):
            if i == r or abs(a.get_element(i, c)) <= EPSILON:
                continue
            d = a.get_element(i, c)/max_val
            a.eo3_r1_multiple_x_add_r2(r, -d, i)
            identify_mat.eo3_r1_multiple_x_add_r2(r, -d, i)

        a.eo2_row_multiple_x(r, 1/max_val)
        identify_mat.eo2_row_multiple_x(r, 1/max_val)
        c += 1

    return identify_mat


def gauss_jordan_elimination(mat: Matrix) -> Matrix:
    a = copy_matrix(mat)
    c = 0
    for r in range(0, a.row_size):
        if c >= a.col_size:
            break

        max_row = r
        max_val = a.get_element(r, c)
        for i in range(r + 1, a.row_size):
            if abs(max_val) < abs(a.get_element(i, c)):
                max_row = i
                max_val = a.get_element(i, c)

        if abs(max_val) <= EPSILON:
            c += 1
            continue

        if max_row != r:
            a.eo1_row_swap(r, max_row)

        for i in range(0, a.row_size):
            if i == r or abs(a.get_element(i, c)) <= EPSILON:
                continue
            d = a.get_element(i, c) / max_val
            a.eo3_r1_multiple_x_add_r2(r, -d, i)

        a.eo2_row_multiple_x(r, 1 / max_val)
        c += 1

    return a


def null_space(mat: Matrix) -> list:
    rref_mat = gauss_jordan_elimination(mat)
    # print(rref_mat)
    ret = []
    for r in range(rref_mat.row_size):
        if abs(rref_mat.get_element(r, r)) > EPSILON:
            continue
        v = Matrix(rref_mat.col_size, 1)
        for i in range(rref_mat.row_size):
            v.set_element(i, 0, -rref_mat.get_element(i, r))
        v.set_element(r, 0, 1)
        ret.append(v)
    return ret


def igen_vec(mat: Matrix, lam: float) -> list:
    return null_space(mat - lam*identify(mat.col_size))


def igen_val(mat: Matrix) -> list:
    if mat.col_size == 2:
        b = -(mat.get_element(0, 0) + mat.get_element(1, 1))
        c = det(mat)
        # print(b, c)
        return [(-b-sqrt(b*b-4*c))/2, (-b+sqrt(b*b-4*c))/2]

    else:
        lams = list(map(eval, input("input lams (split by space) :").split()))
        print(lams)
        return lams


def diagonalization(mat: Matrix) -> list:
    dia_matrix = Matrix(mat.row_size, mat.col_size)
    x_matrix = Matrix(mat.row_size, mat.col_size)
    lams = igen_val(mat)
    idx = 0
    for lam in lams:
        vec_list = igen_vec(mat, lam)
        for i in range(len(vec_list)):
            dia_matrix.set_element(idx+i, idx+i, lam)
            for j in range(vec_list[i].row_size):
                x_matrix.set_element(j, idx+i, vec_list[i].get_element(j, 0))
        idx += len(vec_list)
    return [dia_matrix, x_matrix]
