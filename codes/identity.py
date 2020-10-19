from .linear import LinearCode
import numpy as np


class IdentityCode(LinearCode):
    def __init__(self, block_size):
        G = np.eye(block_size, dtype=np.int32)
        super(IdentityCode, self).__init__(block_size, block_size,
                                           G=G)
