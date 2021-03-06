from .base import BaseChannel
import numpy as np
from codes.bp_code import BPCode


class GaussChannel(BaseChannel):
    """
    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.
    """
    def __init__(self, coder,
                 snr: float = 20.):
        assert isinstance(coder, BPCode), "This channel created only for Belief Propagation algorithms"
        super(GaussChannel, self).__init__(coder)
        self.snr = snr
        self.sigma = 10 ** (-snr / 20)

    def noise_block(self, x):
        x = (-1) ** x
        y = x + np.random.randn(x.shape[0]) * self.sigma
        return y
