from .linear import LinearCode
import numpy as np


class HammingCode(LinearCode):
    def __init__(self, r):
        super(HammingCode, self).__init__(block_size=2 ** r - 1 - r,
                                          code_size=2 ** r - 1)

        self.H = np.zeros()
