from .linear import LinearCode
import numpy as np
import bitarray


class HammingCode(LinearCode):
    def __init__(self, r):
        powers = [2 ** i for i in range(r)]
        P = np.stack([self.int2binary(i, r) for i in range(2 ** r - 1) if i not in powers], axis=-1)  # r x 2**r-1
        H = np.concatenate([
            P,
            np.eye(r),
        ], axis=-1)
        G = np.concatenate([
            np.eye(r),
            P.T,
        ], axis=-1)
        super(HammingCode, self).__init__(block_size=2 ** r - 1 - r,
                                          code_size=2 ** r - 1,
                                          G=G,
                                          H=H)
