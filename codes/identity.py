from .linear import LinearCode
import numpy as np


class IdentityCode(LinearCode):
    def __init__(self, block_size):
        super(IdentityCode, self).__init__(block_size, block_size)
        self.G = np.eye(block_size)
        self.H = np.eye(block_size)
