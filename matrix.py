EPSILON = 1e-7


class Matrix:
    def __init__(self, r: int, c: int):
        self.data = [[0. for x in range(c)] for y in range(r)]
        self.row_size = r
        self.col_size = c

    @staticmethod
    def identify(n: int):
        ret = Matrix(n, n)
        for i in range(n):
            ret.set_element(i, i, 1.)
        return ret

    @staticmethod
    def from_input(r=-1, c=-1):
        if r == -1:
            r = int(input("row_size: "))
            c = int(input("col_size: "))

        elif c == -1:
            c = r

        ret = Matrix(r, c)
        for i in range(r):
            arr = input(f"{i}\'s row : ").split()
            for j in range(c):
                ret.set_element(i, j, float(eval(arr[j])))
        return ret

    @staticmethod
    def from_1d_array(arr: list, is_col_vec=True):
        if is_col_vec:
            ret = Matrix(len(arr), 1)
        else:
            ret = Matrix(1, len(arr))

        for i in range(len(arr)):
            if is_col_vec:
                ret.set_element(i, 0, float(eval(arr[i])))
            else:
                ret.set_element(0, i, float(eval(arr[i])))

        return ret

    @staticmethod
    def from_2d_array(arr: list):
        r = len(arr)
        c = len(arr[0])

        ret = Matrix(r, c)
        for i in range(r):
            for j in range(c):
                ret.set_element(i, j, float(eval(arr[i][j])))

        return ret

    def T(self):
        ret = Matrix(self.row_size, self.col_size)
        for r in range(self.row_size):
            for c in range(self.col_size):
                ret.set_element(c, r, self.get_element(r, c))
        return ret

    def copy(self):
        ret = Matrix(self.row_size, self.col_size)
        for r in range(self.row_size):
            for c in range(self.col_size):
                ret.set_element(r, c, self.get_element(r, c))
        return ret

    def eo1_row_swap(self, r1: int, r2: int) -> None:
        r3 = [ self.get_element(r2, c) for c in range(self.col_size) ]
        for c in range(self.col_size):
            self.set_element(r2, c, self.get_element(r1, c))
            self.set_element(r1, c, r3[c])
    
    def eo2_row_multiple_x(self, r: int, x: float) -> None:
        for c in range(self.col_size):
            self.set_element(r, c, x*self.get_element(r, c))
    
    def eo3_r1_multiple_x_add_r2(self, r1: int, x: float, r2: int) -> None:
        for c in range(self.col_size):
            ori = self.get_element(r2, c)
            delta = self.get_element(r1, c)*x
            self.set_element(r2, c, ori + delta)
    
    def set_element(self, r: int, c: int, val: float) -> None:
        self.data[r][c] = val
    
    def get_element(self, r: int, c: int) -> float:
        return self.data[r][c]

    def __mul__(self, other):
        ret = Matrix(self.row_size, other.col_size)
        for r in range(self.row_size):
            for c in range(other.col_size):
                s = 0.
                for k in range(self.col_size):
                    s += self.get_element(r, k)*other.get_element(k, c)
                ret.set_element(r, c, s)
        return ret
    
    def __rmul__(self, other: float):
        ret = Matrix(self.row_size, self.col_size)
        for r in range(self.row_size):
            for c in range(self.col_size):
                ret.set_element(r, c, self.get_element(r, c)*other)
        return ret

    def __neg__(self):
        ret = Matrix(self.row_size, self.col_size)
        for r in range(self.row_size):
            for c in range(self.col_size):
                ret.set_element(r, c, -self.get_element(r, c))
        return ret

    def __add__(self, other):
        ret = Matrix(self.row_size, self.col_size)
        for r in range(self.row_size):
            for c in range(self.col_size):
                ret.set_element(r, c, self.get_element(r, c) + other.get_element(r, c))
        return ret

    def __sub__(self, other):
        return self+(-other)

    def __eq__(self, other):
        for r in range(self.row_size):
            for c in range(self.col_size):
                if abs(self.get_element(r, c) - other.get_element(r, c)) > EPSILON:
                    return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __gt__(self, other):
        for r in range(self.row_size):
            for c in range(self.col_size):
                if self.get_element(r, c) - other.get_element(r, c) <= EPSILON:
                    return False

        return True

    def __ge__(self, other):
        return self == other or self > other
    
    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return self == other or self < other

    def __pow__(self, y):
        if y == 0:
            return Matrix.identify(self.row_size)
        
        if y % 2:
            return self*(self**(y-1))

        x = (self**(y//2))
        return x*x

    def __repr__(self):

        ret = "\n"
        ret += "---------------------------------------\n"
        ret += f"row_size: {self.row_size}, col_size: {self.col_size}\n"
        ret += "[\n"
        for r in range(self.row_size):
            ret += str(self.data[r])
            ret += "\n"
        ret += "]\n"
        return ret
        