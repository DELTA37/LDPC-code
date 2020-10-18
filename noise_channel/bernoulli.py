from .base import BaseChannel
import numpy as np


class BernoulliChannel(BaseChannel):
    def __init__(self, coder, p=(0.5, 0.5)):
        super(BernoulliChannel, self).__init__(coder=coder)
        self.cdf = np.cumsum(p)
        self.max_n_errors = len(p) - 1
        assert self.max_n_errors <= self.code_size

    def noise_block(self, x):
        assert x.shape[0] == self.code_size
        n_errors = np.searchsorted(self.cdf, np.random.rand())

        s = list(range(self.code_size))
        flip_idx = []
        for i in range(n_errors):
            flip_idx.append(s.pop(np.random.randint(0, self.code_size - i)))
        print(f"n_errors: {n_errors}, flip_idx: {flip_idx}")
        return self.coder.flip_bits(x, flip_idx)
