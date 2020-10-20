from .base import BaseChannel
import numpy as np


class BurstErrorChannel(BaseChannel):
    def __init__(self, coder, max_length, p=0.5):
        super(BurstErrorChannel, self).__init__(coder)
        self.max_length = max_length
        self.p = p
        assert max_length < coder.code_size

    def noise_block(self, x):
        assert x.shape[0] == self.coder.code_size
        l = np.random.randint(0, self.max_length + 1)
        pos = np.random.randint(0, self.coder.code_size - l)
        x[pos: pos + l] = (np.random.rand(l) > self.p)
        x[pos + l - 1] = x[pos] = 1
        print(f"pos: {pos}, l: {l}")
        return x
