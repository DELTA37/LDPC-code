import numpy as np


class poly1d_gf2(np.poly1d):
    def __init__(self, c_or_r, r=False, variable=None):
        super(poly1d_gf2, self).__init__(c_or_r,
                                         r=r,
                                         variable=variable)
        self.to_gf2()

    def to_gf2(self):
        self._coeffs = self._coeffs.astype(np.int32)
        self._coeffs %= 2
        self._coeffs = np.trim_zeros(self._coeffs, 'f')

    def __bool__(self):
        return len(self._coeffs) > 1 or (len(self._coeffs) == 1 and self._coeffs[0])

    def __abs__(self):
        return self

    def __neg__(self):
        return self

    def __mul__(self, other):
        return poly1d_gf2(super(poly1d_gf2, self).__mul__(other).coeffs)

    def __add__(self, other):
        return poly1d_gf2(super(poly1d_gf2, self).__add__(other).coeffs)

    def __sub__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return poly1d_gf2(super(poly1d_gf2, self).__radd__(other).coeffs)

    def __pow__(self, power):
        return poly1d_gf2(super(poly1d_gf2, self).__pow__(power).coeffs)

    def euclid_div(self, other: 'poly1d_gf2'):
        if self.order < other.order:
            return poly1d_gf2([0]), self
        h = self.create_basis(self.order - other.order)
        r = self + h * other
        h1, r1 = r.euclid_div(other)
        return h + h1, r1

    @staticmethod
    def create_basis(order):
        c = np.zeros(order + 1)
        c[0] = 1
        return poly1d_gf2(c)

    @staticmethod
    def zeros():
        return poly1d_gf2([0])

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d_gf2(%s)" % vals

    def flipud(self):
        return poly1d_gf2(np.flipud(self._coeffs))

    def to_code(self, zfill):
        c = np.array(self._coeffs)
        if c.shape[0] < zfill:
            c = np.concatenate([np.zeros(zfill - c.shape[0], dtype=np.int32), c])
        c = np.flipud(c)
        return c

    @staticmethod
    def from_code(code):
        return poly1d_gf2(np.flipud(code))

    def __div__(self, other):
        q, r = self.euclid_div(other)
        assert not r
        return q

    def __floordiv__(self, other):
        return self.euclid_div(other)[0]
