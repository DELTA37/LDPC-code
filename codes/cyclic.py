from .polynomial import PolynomialCode
from .poly_gf2 import poly1d_gf2


class CyclicCode(PolynomialCode):
    def __init__(self, block_size, q: poly1d_gf2):
        super(CyclicCode, self).__init__(block_size, q=q)
