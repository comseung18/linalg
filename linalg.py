from matrix import Matrix
from matrix import EPSILON
from math import *
import cmath

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


def normalized(mat: Matrix) -> Matrix:
    ret = copy_matrix(mat)
    for c in range(ret.col_size):
        v = ret.get_col_vec(c)
        v = (1/length(v))*v
        ret.set_col_vec(c, v)
    return ret


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
    r = 0
    while r < a.row_size and c < a.col_size:

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
        r += 1
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
    r = 0
    while r < a.row_size and c < a.col_size:

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
        r += 1

    return a


def get_no_reading_one_cols(mat: Matrix) -> list:
    r = 0
    c = 0
    ret = []
    while r < mat.row_size and c < mat.col_size:
        if abs(mat.get_element(r, c)) >= EPSILON:
            r += 1
            c += 1
        else:
            ret.append(c)
            c += 1

    return ret


def get_reading_one_col_by_row(mat: Matrix, row: int) -> int:
    for c in range(mat.col_size):
        if abs(mat.get_element(row, c)-1) <= EPSILON:
            return c
    return -1


def null_space(mat: Matrix) -> list:
    rref_mat = gauss_jordan_elimination(mat)
    if rref_mat.row_size < rref_mat.col_size:
        tmp = Matrix(rref_mat.col_size, rref_mat.col_size)
        for r in range(rref_mat.row_size):
            for c in range(rref_mat.col_size):
                tmp.set_element(r, c, rref_mat.get_element(r, c))
        rref_mat = tmp

    zero_col_cnt = 0
    col_variable_index_mapper = dict()
    rref_mat_temp = Matrix(rref_mat.row_size, 0)

    for c in range(rref_mat.col_size):
        if rref_mat.get_col_vec(c) == zeros(rref_mat.row_size, 1):
            zero_col_cnt += 1

    k = 0
    j = 0

    for c in range(rref_mat.col_size):
        if rref_mat.get_col_vec(c) == zeros(rref_mat.row_size, 1):
            col_variable_index_mapper[rref_mat.col_size - zero_col_cnt + j] = c
            j += 1
        else:
            rref_mat_temp.append_col_vec(rref_mat.get_col_vec(c))
            col_variable_index_mapper[k] = c
            k += 1

    for i in range(zero_col_cnt):
        rref_mat_temp.append_col_vec(zeros(rref_mat.row_size, 1))

    rref_mat = rref_mat_temp
    ret = []

    none_reading_one_cols = get_no_reading_one_cols(rref_mat)


    for c in none_reading_one_cols:
        v = Matrix(rref_mat.col_size, 1)
        for i in range(c):
            r = get_reading_one_col_by_row(rref_mat, i)
            if r == -1:
                continue
            v.set_element(r, 0, -rref_mat.get_element(i, c))
        v.set_element(c, 0, 1)

        g = Matrix(rref_mat.col_size, 1)

        for i in range(rref_mat.col_size):
            g.set_element(col_variable_index_mapper[i], 0, v.get_element(i, 0))

        ret.append(g)

    return ret


def row_space(mat: Matrix) -> list:
    rref_mat = rref(mat)
    ret = []
    r = 0
    c = 0
    while r < rref_mat.row_size and c < rref_mat.col_size:
        if abs(rref_mat.get_element(r, c)) > EPSILON:
            ret.append(rref_mat.get_row_vec(r))
            r += 1
            c += 1
        else:
            c += 1

    return ret


def col_space(mat: Matrix) -> list:
    rref_mat = rref(mat)
    ret = []
    r = 0
    c = 0
    while r < rref_mat.row_size and c < rref_mat.col_size:
        if abs(rref_mat.get_element(r, c)) > EPSILON:
            ret.append(mat.get_col_vec(c))
            r += 1
            c += 1
        else:
            c += 1

    return ret


def igen_vec(mat: Matrix, lam: float) -> list:
    return null_space(mat - lam*identify(mat.col_size))


def igen_val(mat: Matrix) -> list:
    if mat.col_size == 2:
        b = -(mat.get_element(0, 0) + mat.get_element(1, 1))
        c = det(mat)
        d = b*b-4*c
        if d > 0:
            return [(-b-sqrt(d))/2, (-b+sqrt(d))/2]
        elif d == 0:
            return [-b/2]
        return [(-b-cmath.sqrt(d))/2, (-b+cmath.sqrt(d))/2]
    else:
        lams = list(map(eval, input("input lams (split by space) :").split()))
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


def proj(w: Matrix, v: Matrix) -> Matrix:
    ret = Matrix(w.row_size, 1)
    for c in range(w.col_size):
        u = w.get_col_vec(c)
        ret += (dot(u, v)/dot(u, u))*u
    return ret


def perp(w: Matrix, v:Matrix) -> Matrix:
    return v - proj(w, v)


def gram_schmidt_process(x: Matrix, normalize=True) -> Matrix:
    w = Matrix(x.row_size, 0)
    for c in range(x.col_size):
        v = x.get_col_vec(c)
        if c:
            v = v - proj(w, v)

        if normalize:
            w.append_col_vec(normalized(v))
        else:
            w.append_col_vec(v)

    return w


def modified_qr_decomposition(a: Matrix) -> list:

    x = a.get_col_vec(0)

    y = zeros(x.row_size, 1)
    y.set_element(0, 0, sqrt(dot(x, x)))

    if x != y:
        u = normalized(x-y)
        q = householder_matrix(u)
    else:
        q = identify(x.row_size)

    r = q*a
    acc_q = copy_matrix(q)
    for c in range(1, min(a.col_size, a.row_size)-1):
        x = r.get_sub_col_vec(c, c)
        y = zeros(x.row_size, 1)
        y.set_element(0, 0, sqrt(dot(x, x)))

        if x != y:
            u = normalized(x-y)
            p = householder_matrix(u)
        else:
            p = identify(x.row_size)

        q = identify(a.row_size)
        for i in range(c, a.row_size):
            for j in range(c, a.row_size):
                q.set_element(i, j, p.get_element(i-c, j-c))
        r = q * r
        acc_q = acc_q * q
    return [acc_q, r]


def qr_decomposition(x: Matrix) -> list:
    q = gram_schmidt_process(x)
    r = transpose(q)*x
    return [q, r]


def householder_matrix(u: Matrix) -> Matrix:
    return identify(u.row_size) - 2.0*(u*transpose(u))


def check_ele_in_list(li: list, ele: float):
    if len(li) == 0: return False
    for i in range(len(li)):
        if abs(li[i] - ele) <= EPSILON:
            return True
    return False


def qr_algorithm(a: Matrix, it: int = 100) -> list:
    m = copy_matrix(a)
    for i in range(it):
        q, r = modified_qr_decomposition(m)
        m = r*q
    ret = list()
    for i in range(m.col_size):
        if not check_ele_in_list(ret, m.get_element(i, i)):
            ret.append(m.get_element(i, i))
    return ret


def spectral_decomposition(a: Matrix) -> list:
    lams = qr_algorithm(a)
    if not check_lams(a, lams):
        lams = list(map(eval, input("input lams (split by space) :").split()))

    q = Matrix(a.row_size, 0)
    d = identify(a.col_size)
    i = 0
    for lam in lams:
        ker: list = igen_vec(a, lam)
        ker: Matrix = Matrix.from_col_vec_list(ker)
        ker: Matrix = qr_decomposition(ker)[0]
        for c in range(ker.col_size):
            q.append_col_vec(ker.get_col_vec(c))
            d.set_element(i+c, i+c, lam)
        i += ker.col_size

    return [d, q]


def check_lams(a: Matrix, lams: list) -> bool:
    cnt = 0
    for lam in lams:
        cnt += len(igen_vec(a, lam))
    return cnt == a.col_size
