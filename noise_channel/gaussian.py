from .base import BaseChannel
import numpy as np


class GaussChannel(BaseChannel):
    """
    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.
    """
    def __init__(self, coder,
                 snr: float = 20.):
        super(GaussChannel, self).__init__(coder)
        self.snr = snr
        self.sigma = 10 ** (-snr / 20)

    def noise_block(self, x):
        return x + np.random.randn(x.shape[0]) * self.sigma
